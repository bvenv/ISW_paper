#!/usr/bin/env python3
"""
Compute pseudo-C_ell for DESI g × Planck T per redshift bin using NaMaster/MASTER.

Inputs
- DESI delta maps: maps/desi/{TRACER}/nside{NSIDE}/delta_{TRACER}_z{ibin}.fits.gz
- Planck T map:    maps/planck/nside{NSIDE}/cmb_T_{LABEL}.fits.gz
- Joint mask:      masks/planck_desi_joint_nside{NSIDE}.fits.gz

Outputs (per tracer/bin)
- spectra/desi/gT_{TRACER}_z{ibin}_lmax{L}.npz
  keys: ell, cl, ell_edges_lo, ell_edges_hi, nside, lmin, lmax, delta_ell,
        tracer, ibin, cmb_label, mask_path, wsp_path, apotype, apod_deg,
        pixwin_deconvolved=True
"""
import argparse
import logging
from pathlib import Path
import os
import yaml
import numpy as np
import healpy as hp

try:
    import pymaster as nmt
except Exception as e:
    raise RuntimeError("pymaster (NaMaster) is required. Install with `pip install pymaster` "
                       "or your env's package manager.") from e

LOGGER = logging.getLogger("compute_crosscls")


# ---------- CLI / config ----------
def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="Path to YAML config (configs/desi.yml)")
    p.add_argument("--nside", type=int, default=1024, help="NSIDE for inputs/outputs")
    p.add_argument("--ell-min", type=int, default=2, help="Minimum ell (inclusive)")
    p.add_argument("--ell-max", type=int, default=150, help="Maximum ell (inclusive or nearest below)")
    p.add_argument("--delta-ell", type=int, default=10, help="Bin width Δℓ")
    p.add_argument("--apod-deg", type=float, default=1.0, help="Cosine apodization scale (degrees)")
    p.add_argument("--apotype", choices=["C1", "C2"], default="C2", help="NaMaster apodization type")
    p.add_argument("--cmb-label", default="SMICA", help="Which CMB map label to use (SMICA, Commander, SEVEM)")
    p.add_argument("--cmb-field", choices=["T", "kappa"], default="T",
                   help="Cross delta_g with the CMB temperature (T → gT) or lensing kappa (kappa → kg). "
                        "kappa uses maps/planck/nside{N}/kappa.fits.gz and the kappa joint mask.")
    p.add_argument("--tracers", nargs="+", default=["BGS", "LRG", "ELG", "QSO"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument(
    "--apodizer",
    choices=["healpy-gauss", "nmt", "none"],
    default="healpy-gauss",
    help="How to apodize the joint mask before building NmtField")
    p.add_argument("--plot", action="store_true",
               help="Plot spectra for any bins computed in this run")
    p.add_argument("--plot-only", action="store_true",
                help="Only plot existing NPZs (skip computing)")
    p.add_argument("--plot-dell", action="store_true",
                help="Plot D_ell = ell(ell+1)C_ell/2pi instead of C_ell")
    p.add_argument("--plots-outdir", default=None,
                help="Directory for plots (default: <spectra_root>/plots)")
    p.add_argument("--ylim", nargs=2, type=float, default=None,
                help="Optional y-limits for the plot (min max)")


    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- helpers ----------
def _paths(cfg, nside, tracer, ibin, cmb_label, cmb_field="T"):
    maps_root = Path(cfg["paths"]["maps"])
    masks_root = Path(cfg["paths"]["masks"])
    spectra_root = Path(cfg["paths"]["spectra"]) / "desi"

    delta_map = maps_root / "desi" / tracer / f"nside{nside}" / f"delta_{tracer}_z{ibin}.fits.gz"
    if cmb_field == "kappa":
        cmb_map    = maps_root / "planck" / f"nside{nside}" / "kappa.fits.gz"
        joint_mask = masks_root / f"planck_desi_kappa_joint_nside{nside}.fits.gz"
    else:
        cmb_map    = maps_root / "planck" / f"nside{nside}" / f"cmb_T_{cmb_label}.fits.gz"
        joint_mask = masks_root / f"planck_desi_joint_nside{nside}.fits.gz"
    spectra_root.mkdir(parents=True, exist_ok=True)

    return delta_map, cmb_map, joint_mask, spectra_root


def _load_mask_ring(path, nside):
    # Map may be byte or float, RING or NESTED; return float mask in [0,1], RING, at target nside
    m, hdr = hp.read_map(path, dtype=np.float64, h=True, verbose=False)
    ordering = None
    for (k, v) in hdr:
        if k == "ORDERING":
            ordering = str(v).strip().upper()
            break
    if ordering == "NESTED":
        m = hp.reorder(m, n2r=True)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside, order_in="RING", order_out="RING", power=None)
    m = np.clip(m, 0.0, 1.0)
    return m


def _load_map_ring(path, nside):
    m = hp.read_map(path, dtype=np.float64, verbose=False)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside, order_in="RING", order_out="RING", power=None)
    return m


def _apodize(mask, apod_deg, apotype, method):
    """
    Apodize a binary mask in a robust way.
    - 'healpy-gauss': Gaussian Smooth (FWHM=apod_deg), then clip inside footprint.
    - 'nmt': call NaMaster's mask_apodization (kept as an option).
    - 'none': return the raw mask.
    """
    if apod_deg <= 0 or method == "none":
        return mask

    if method == "healpy-gauss":
        # Gaussian FWHM in radians
        fwhm_rad = np.deg2rad(apod_deg)
        # Smooth the *binary* mask; hp.smoothing returns float in [0,1] (with small ringing)
        sm = hp.smoothing(mask.astype(np.float64), fwhm=fwhm_rad, verbose=False)
        # Ensure no weight leaks outside the original support
        sm *= (mask > 0).astype(np.float64)
        # Clip to [0,1] just in case of minor overshoots from smoothing
        sm = np.clip(sm, 0.0, 1.0)
        return sm

    if method == "nmt":
        # Keep as a fallback; guard against too-small radii
        nside = hp.get_nside(mask)
        pix_deg = np.degrees(hp.nside2resol(nside))
        apos_deg = max(apod_deg, 3.0 * pix_deg)
        try:
            return nmt.mask_apodization(mask.astype(np.float64),
                                        np.deg2rad(apos_deg),
                                        apotype=apotype)
        except Exception:
            # Final fallback: try a bit larger or 'Smooth'
            try:
                return nmt.mask_apodization(mask.astype(np.float64),
                                            np.deg2rad(apos_deg * 1.5),
                                            apotype=apotype)
            except Exception:
                return nmt.mask_apodization(mask.astype(np.float64),
                                            np.deg2rad(apos_deg * 1.5),
                                            apotype="Smooth")

    # default safety
    return mask



def _make_bin(nside, lmin, lmax, d_ell):
    """
    Build NaMaster bins covering the FULL harmonic range up to 3*nside-1 with
    uniform width Δℓ=d_ell. lmin/lmax are unused here; selection happens later
    in _compute_binned_cls after decoupling.
    """
    lmax_full = 3 * nside - 1

    # Prefer the simplest, version-proof constructor
    if hasattr(nmt.NmtBin, "from_nside_linear"):
        # Old/new APIs both accept (nside, nlb) without lmin/lmax
        return nmt.NmtBin.from_nside_linear(nside, nlb=d_ell)

    # Fallback: explicit edges over the full range
    edges = np.arange(0, lmax_full + 1, d_ell, dtype=int)
    if edges[-1] != lmax_full + 1:
        edges = np.append(edges, lmax_full + 1)

    if hasattr(nmt.NmtBin, "from_edges"):
        return nmt.NmtBin.from_edges(edges[:-1], edges[1:])

    # Last-resort: explicit ell lists
    ell_list = [np.arange(edges[i], edges[i + 1], dtype=int)
                for i in range(len(edges) - 1)]
    return nmt.NmtBin(ell_list)




def _workspace_path(root: Path, nside, d_ell, apod_deg, apotype, cmb_field="T", tracer=""):
    wdir = root / "workspaces"
    wdir.mkdir(parents=True, exist_ok=True)
    lmax_full = 3 * nside - 1
    # The coupling matrix depends on the mask = joint × finite(delta_tracer); different
    # tracers have different footprints, and the kappa-joint differs from the T-joint.
    ftag = "" if cmb_field == "T" else f"_{cmb_field}"
    ttag = f"_{tracer}" if tracer else ""
    tag = f"n{nside}_l0-{lmax_full}_d{d_ell}_apo{apod_deg:.2f}{apotype}{ftag}{ttag}"
    return wdir / f"wsp_{tag}.fits"



def _build_fields(delta_map, cmb_map, joint_mask, apod_deg, apotype, apod_method):
    nside = hp.get_nside(joint_mask)
    m_delta = _load_map_ring(delta_map, nside)
    m_cmb   = _load_map_ring(cmb_map, nside)

    finite_delta = np.isfinite(m_delta) & (m_delta != hp.UNSEEN)
    finite_cmb   = np.isfinite(m_cmb) & (m_cmb != hp.UNSEEN)

    msk_raw = joint_mask * finite_delta.astype(float) * finite_cmb.astype(float)
    fsky_raw = (msk_raw > 0).mean()
    LOGGER.info("  f_sky (raw mask before apod) = %.3f", fsky_raw)

    if fsky_raw <= 0:
        raise RuntimeError("Joint × finite mask is empty (f_sky=0). Check inputs/paths.")

    msk_apo = _apodize(msk_raw, apod_deg, apotype, method=apod_method)
    fsky_apo = (msk_apo > 0).mean()
    LOGGER.info("  f_sky (after apodization) = %.3f  [apod_deg=%g, apotype=%s]",
                fsky_apo, apod_deg, apotype)

    f_g = nmt.NmtField(msk_apo, [m_delta])   # spin-0
    f_T = nmt.NmtField(msk_apo, [m_cmb])     # spin-0
    return f_g, f_T, msk_apo, fsky_apo


def _compute_binned_cls(f_g, f_T, binning, wsp_path, lmin_sel, lmax_sel):
    # Workspace (coupling matrix)
    wsp = nmt.NmtWorkspace()
    if wsp_path.exists():
        wsp.read_from(str(wsp_path))
    else:
        wsp.compute_coupling_matrix(f_g, f_T, binning)
        wsp.write_to(str(wsp_path))

    # Coupled / decoupled spectra
    cl_coup = nmt.compute_coupled_cell(f_g, f_T)   # list; spin-0×0 → one array
    cl_dec  = wsp.decouple_cell(cl_coup)[0]

    # Bin meta (robust across versions)
    if hasattr(binning, "get_effective_ells"):
        ells_eff = binning.get_effective_ells()
        nband=binning.get_n_bands()
        ells_low=np.zeros_like(ells_eff)
        ells_high=np.zeros_like(ells_eff)
        for i in range(nband):
            ells_low[i]=binning.get_ell_min(i)
            ells_high[i]=binning.get_ell_max(i)

    else:
        raise ValueError

    # Select analysis range *after* decoupling
    sel = (ells_eff >= max(2, lmin_sel)) & (ells_eff <= lmax_sel)
    return ells_eff[sel], ells_low[sel], ells_high[sel], cl_dec[sel], wsp


def _plots_dir(spectra_root: Path, override: str | None):
    out = Path(override) if override else (spectra_root / "plots")
    out.mkdir(parents=True, exist_ok=True)
    return out

def _plot_npz(npz_path: Path, outdir: Path, use_dell: bool = False, ylim=None):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    d = np.load(npz_path)
    ell = d["ell"]; cl = d["cl"]
    tracer = str(d.get("tracer", "TRACER"))
    ibin   = int(d.get("ibin", -1))
    lmin   = int(d.get("lmin", ell.min() if ell.size else 0))
    lmax   = int(d.get("lmax", ell.max() if ell.size else 0))
    nside  = int(d.get("nside", -1))
    cmb    = str(d.get("cmb_label", "CMB"))
    apotype = str(d.get("apotype", ""))
    apod_deg = float(d.get("apod_deg", 0.0))

    y = cl
    ylab = r"$C_\ell^{gT}$ [$\mu$K]"
    tag = "Cl"
    if use_dell:
        fac = ell * (ell + 1) / (2.0 * np.pi)
        y = fac * cl
        ylab = r"$D_\ell^{gT} \equiv \ell(\ell+1)C_\ell^{gT}/2\pi$ [$\mu$K]"
        tag = "Dell"

    # Try to plot diagonal errors if present
    yerr = None
    if "cov" in d.files:
        import numpy as np
        try:
            cov = d["cov"]
            yerr = np.sqrt(np.diag(cov))
        except Exception:
            pass
    elif "cl_err" in d.files:
        try:
            yerr = d["cl_err"]
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if yerr is not None:
        ax.errorbar(ell, y, yerr=yerr if not use_dell else yerr * (ell * (ell + 1) / (2*np.pi)),
                    fmt="o", ms=4, alpha=0.9, capsize=2)
    else:
        ax.plot(ell, y, "o-", ms=4, lw=1.1)

    ax.axhline(0.0, ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(ylab)
    ax.loglog()
    ttl = f"{tracer} z{ibin} × {cmb}  (NSIDE={nside}, ℓ={lmin}–{lmax}, apod={apod_deg}° {apotype})"
    ax.set_title(ttl, fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, ls=":", alpha=0.4)

    base = f"gT_{tracer}_z{ibin}_lmax{lmax}_{tag}"
    out_png = outdir / f"{base}.png"
    out_pdf = outdir / f"{base}.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    LOGGER.info("Plotted → %s", out_png)



# ---------- per-bin driver ----------
def compute_for_bin(cfg: dict, tracer: str, ibin: int, nside: int,
                    ell_min: int, ell_max: int, delta_ell: int,
                    apod_deg: float, apotype: str,apod_method: str,
                    cmb_label: str, out_npz: Path, wsp_cache_root: Path,
                    cmb_field: str = "T"):

    delta_map, cmb_map, joint_mask_path, spectra_root = _paths(cfg, nside, tracer, ibin, cmb_label, cmb_field)

    # Existence checks
    for p in (delta_map, cmb_map, joint_mask_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    # Load joint mask (float [0,1], RING, at nside)
    joint_mask = _load_mask_ring(joint_mask_path, nside)

    # Build fields
    f_g, f_T, msk_apo, fsky = _build_fields(delta_map, cmb_map, joint_mask,
                                        apod_deg, apotype, apod_method=apod_method)

    LOGGER.info("Tracer=%s zbin=%d f_sky(apodized)=%.3f", tracer, ibin, fsky)

    # Binning and workspace
    binning = _make_bin(nside, ell_min, ell_max, delta_ell)
    wsp_path = _workspace_path(wsp_cache_root, nside, delta_ell, apod_deg, apotype, cmb_field, tracer)


    # Spectra
    ell, elo, ehi, cl, wsp = _compute_binned_cls(
        f_g, f_T, binning, wsp_path, lmin_sel=ell_min, lmax_sel=ell_max
    )

    # Deconvolve the HEALPix pixel window of the galaxy map (δ_g carries W_pix; the CMB T/κ
    # side is ~1 at these ℓ). One power of W_pix for a δ_g × CMB cross. (~1 at ℓ≤150.)
    pw = hp.pixwin(nside, lmax=3 * nside - 1)
    pwb = np.array([pw[int(a):int(b) + 1].mean() for a, b in zip(elo, ehi)])
    cl = cl / pwb


    # Save
    np.savez(
        out_npz,
        ell=ell,
        cl=cl,
        ell_edges_lo=elo,
        ell_edges_hi=ehi,
        nside=nside,
        lmin=ell_min,
        lmax=ell_max,
        delta_ell=delta_ell,
        tracer=tracer,
        ibin=ibin,
        cmb_label=cmb_label,
        cmb_field=cmb_field,
        mask_path=str(joint_mask_path),
        wsp_path=str(wsp_path),
        apotype=apotype,
        apod_deg=apod_deg,
        pixwin_deconvolved=True,
    )
    LOGGER.info("Wrote spectrum → %s", out_npz)

    # plot immediately if requested
    if cfg.get("_PLOT_NOW", False):  # set in main()
        plots_dir = _plots_dir(spectra_root, cfg.get("_PLOTS_OUT"))
        _plot_npz(out_npz, plots_dir, use_dell=cfg.get("_PLOT_DELL", False), ylim=cfg.get("_PLOT_YLIM"))


    


# ---------- main ----------
def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)

    # Dirs
    spectra_root = Path(cfg["paths"]["spectra"]) / "desi"
    spectra_root.mkdir(parents=True, exist_ok=True)
    wsp_cache_root = spectra_root / "workspaces"

    plots_dir = _plots_dir(Path(cfg["paths"]["spectra"]) / "desi", args.plots_outdir)
    # stash lightweight switches in cfg to avoid threading many params
    cfg["_PLOT_NOW"]  = bool(args.plot and not args.plot_only)
    cfg["_PLOTS_OUT"] = str(plots_dir) if plots_dir else None
    cfg["_PLOT_DELL"] = bool(args.plot_dell)
    cfg["_PLOT_YLIM"] = tuple(args.ylim) if args.ylim else None

    # Output prefix: gT_ for temperature cross, kg_ for lensing cross
    prefix = "kg" if args.cmb_field == "kappa" else "gT"

    if args.plot_only:
        # Iterate all requested tracers/bins and plot if NPZ exists
        for tracer in args.tracers:
            bins = cfg["desi"]["bins"].get(tracer, [])
            for ibin, _ in enumerate(bins, start=1):
                npz = (Path(cfg["paths"]["spectra"]) / "desi" /
                       f"{prefix}_{tracer}_z{ibin}_lmax{args.ell_max}.npz")
                if npz.exists():
                    _plot_npz(npz, plots_dir, use_dell=args.plot_dell, ylim=args.ylim)
                else:
                    LOGGER.warning("Missing spectrum (skip): %s", npz)
        return



    for tracer in args.tracers:
        bins = cfg["desi"]["bins"].get(tracer, [])
        if not bins:
            LOGGER.warning("No bins configured for tracer %s; skipping.", tracer)
            continue
        for ibin, _ in enumerate(bins, start=1):
            out = spectra_root / f"{prefix}_{tracer}_z{ibin}_lmax{args.ell_max}.npz"
            if out.exists() and not args.overwrite:
                LOGGER.info("Skip existing %s", out)
                continue
            compute_for_bin(
                cfg, tracer, ibin,
                nside=args.nside,
                ell_min=args.ell_min,
                ell_max=args.ell_max,
                delta_ell=args.delta_ell,
                apod_deg=args.apod_deg,
                apotype=args.apotype,
                cmb_label=args.cmb_label,
                cmb_field=args.cmb_field,
                out_npz=out,
                wsp_cache_root=wsp_cache_root,
                apod_method=args.apodizer
            )

        


if __name__ == "__main__":
    main()
