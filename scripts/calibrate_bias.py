#!/usr/bin/env python3
"""
Estimate linear galaxy bias per DESI tomographic bin from autos C_ell.

Model per band b:   C_b^{gg, data}  ≈  x * T_b  +  N
  where x = b^2, T_b = theory at unit bias (band-averaged), and N is a flat
  shot term (fitted).  Saves {b_i, sigma_b_i} as JSON priors.

Inputs (from earlier steps):
  - DESI delta maps: maps/desi/<TRACER>/nside{NSIDE}/delta_<TRACER>_z{IBIN}.fits.gz
  - LSS mask (default): masks/desi_lssmask_<TRACER>_nside{NSIDE}.fits.gz
    (or --use-joint-mask to use masks/planck_desi_joint_nside{NSIDE}.fits.gz)

Theory:
  - Uses clank.* modules (or local fallbacks like in mt_fnl_cosmo_mcmc.py)
  - dN/dz = top-hat on [zmin,zmax] unless --nz-dir provides histograms.

Binning:
  - Full-range NaMaster bins (0..3*NSIDE-1, Δℓ=--delta-ell).
  - We *compute/decouple* on the full range and then *select* [ℓ_min, ℓ_max].

Diagnostics:
  - Optional quick plot per bin with data, best-fit model, and N.
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import healpy as hp
import pymaster as nmt
import yaml
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# --- clank-v4 theory engine (run in the clankv4_dev env) ---------------------
from clank.cosmology import Cosmology
from clank.cosmology.matter_power import MatterPowerSpectrum
from clank.tracers.number_counts_tracer import NumberCountsTracer
from clank.tracers.bias import ConstantBias
from clank.observables.angular_cl import angular_cl


LOGGER = logging.getLogger("calibrate_bias")


# ------------------------------ CLI ------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config (same one you used before)")
    p.add_argument("--tracers", nargs="+", default=None, help="Subset to process (default: all in config)")
    p.add_argument("--nside", type=int, required=True)
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--bias-lmin", type=int, default=30,
                   help="Minimum ell used in the b^2/N regression. Low-ell bandpowers "
                        "are dominated by large-scale imaging systematics and bias the fit "
                        "(can drive b^2<0); excluding them stabilises the bias estimate.")
    p.add_argument("--delta-ell", type=int, default=10)
    p.add_argument("--apod-deg", type=float, default=1.0, help="Gaussian apodization (healpy.smoothing)")
    p.add_argument("--use-joint-mask", action="store_true",
                   help="Use masks/planck_desi_joint_nside{NSIDE}.fits.gz instead of tracer LSS mask")
    p.add_argument("--apply-pixwin", action="store_true", help="Multiply theory by pixel window W_l^2")
    p.add_argument("--nz-dir", type=str, default=None,
                   help="If given, use histograms here: z_edges.npy + nz_<TRACER>_z{IBIN}.npy")
    p.add_argument("--outfile", default="results/desi_bias_priors.json")
    p.add_argument("--plot", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("--make-summary-plots", action="store_true",
               help="After running all bins, make n(z) and b(z) summary plots")
    p.add_argument("--plots-outdir", default="results/plots",
                help="Directory for summary plots (n(z), b(z))")

    return p.parse_args()


def setup_logging(v: int):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ------------------------------ I/O ------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _delta_map_path(cfg, tracer: str, nside: int, ibin: int) -> Path:
    root = Path(cfg["paths"]["maps"]) / "desi" / tracer / f"nside{nside}"
    return root / f"delta_{tracer}_z{ibin}.fits.gz"


def _mask_path(cfg, tracer: str, nside: int, use_joint: bool) -> Path:
    if use_joint:
        return Path(cfg["paths"]["masks"]) / f"planck_desi_joint_nside{nside}.fits.gz"
    else:
        return Path(cfg["paths"]["masks"]) / f"desi_lssmask_{tracer}_nside{nside}.fits.gz"
    
def make_cosmology_from_config(cfg):
    """Build a Cosmology() from cfg['theory']['cosmo']."""
    c = (cfg.get("theory") or {}).get("cosmo") or {}

    # Accept either H0 or h
    if "h" in c:
        h = float(c["h"])
    else:
        H0 = float(c.get("H0", 67.36))
        h = H0 / 100.0

    # Physical densities -> fractional densities
    ombh2 = float(c.get("ombh2", 0.02237))
    omch2 = float(c.get("omch2", 0.1200))
    Omega_b = ombh2 / (h*h)
    Omega_c = omch2 / (h*h)

    n_s = float(c.get("ns", c.get("n_s", 0.9649)))
    As  = float(c.get("As", 2.1e-9))
    tau = float(c.get("tau", 0.054))  # Planck 2018-ish default

    LOGGER.info("Cosmo from config: h=%.4f  Ωb=%.4f  Ωc=%.4f  n_s=%.4f  τ=%.3f  As=%.3e",
                h, Omega_b, Omega_c, n_s, tau, As)

    return Cosmology(h=h, Omega_c=Omega_c, Omega_b=Omega_b, n_s=n_s, tau=tau, As=As)



# ------------------------------ binning & mask -------------------------------
from compute_crosscls import _make_bin


def _bin_meta(binning: nmt.NmtBin, nside: int, d_ell: int):
    """
    Deterministic bin description matching from_nside_linear(nside, nlb=d_ell).
    Returns effective ells, low/high edges, and number of multipoles per bin.
    """
    lmax_full = 3 * nside - 1
    edges = np.arange(0, lmax_full + 1, d_ell, dtype=int)
    if edges[-1] != lmax_full + 1:
        edges = np.append(edges, lmax_full + 1)

    ells_low  = edges[:-1]
    ells_high = edges[1:]             # exclusive upper edge
    nell      = ells_high - ells_low  # count of integer ℓ in each bin
    # Midpoint as effective ℓ (works well for smooth spectra)
    ells_eff  = 0.5 * (ells_low + ells_high - 1)

    return ells_eff.astype(float), ells_low, ells_high, nell



def _apodize_gauss(mask: np.ndarray, apod_deg: float) -> np.ndarray:
    fwhm_rad = np.deg2rad(apod_deg)   # we’ll pass this to hp.smoothing as FWHM
    # hp.smoothing argument is FWHM; but we logged sigma earlier; here just pass FWHM.
    sm = hp.smoothing(mask.astype(float), fwhm=fwhm_rad, verbose=False)
    sm = np.clip(sm, 0.0, 1.0)
    return sm


def _build_field(delta_map: np.ndarray, mask: np.ndarray, apod_deg: float) -> tuple[nmt.NmtField, np.ndarray, float]:
    d = np.array(delta_map, float)
    # Pixels with no galaxy data are stored as hp.UNSEEN (~-1.6e30). They MUST be
    # dropped from the mask and zeroed in the map: Gaussian apodization spreads the
    # mask slightly outward, and mask×UNSEEN would otherwise blow the auto C_ell up
    # to ~1e57 (cf. compute_crosscls._build_fields, which folds finite() into the mask).
    good = np.isfinite(d) & (d != hp.UNSEEN)

    msk = np.array(mask, float)
    msk[np.logical_not(np.isfinite(msk))] = 0.0
    msk[msk < 0] = 0.0
    msk = np.clip(msk, 0.0, 1.0)
    msk = msk * good  # no weight where there is no delta data

    msk_apo = _apodize_gauss(msk, apod_deg) if apod_deg > 0 else msk
    d = np.where(good, d, 0.0)  # zero bad pixels so apodization leakage can't blow up

    fsky2 = float(np.mean(msk_apo**2))        # for Gaussian variance formulas
    field = nmt.NmtField(msk_apo, [d])
    return field, msk_apo, fsky2


# ------------------------------ theory template ------------------------------
def _get_dndz_for_bin(cfg, tracer: str, ibin: int, nz_dir: Path | None, zmin: float, zmax: float):
    """Return (z, n(z)) normalized. Prefer external hist if provided; else top-hat."""
    if nz_dir is not None:
        ze = nz_dir / "z_edges.npy"
        zn = nz_dir / f"nz_{tracer}_z{ibin}.npy"
        if ze.exists() and zn.exists():
            edges = np.load(ze); hist = np.load(zn).astype(float)
            zc = 0.5 * (edges[:-1] + edges[1:])
            if hist.sum() > 0: hist /= np.trapezoid(hist, zc)
            return zc, hist
    # fallback: top-hat inside [zmin,zmax]
    z = np.linspace(zmin, zmax, 256)
    n = np.ones_like(z) / max(zmax - zmin, 1e-6)
    n /= np.trapezoid(n, z)
    return z, n


def _theory_unitbias_binned(cosmo: Cosmology,
                            mp: MatterPowerSpectrum,
                            z: np.ndarray, nz: np.ndarray,
                            nside: int,
                            ells_low: np.ndarray,
                            ells_high: np.ndarray,
                            apply_pixwin: bool) -> np.ndarray:
    """
    Band-average C_ell^{gg} (unit bias) over integer ℓ in [low, high) with (2ℓ+1) weights.
    """
    lmax_full = 3 * nside - 1
    lmin_needed = max(2, int(ells_low[0]))
    lmax_needed = min(int(ells_high[-1]) - 1, lmax_full)
    fine_ells = np.arange(lmin_needed, lmax_needed + 1, dtype=int)

    tr = NumberCountsTracer(cosmo, dndz=(z, nz), bias=ConstantBias(1.0))
    Cl_fine = np.asarray(angular_cl(cosmo, tr, tr, fine_ells, mode="limber", mp=mp), float)

    Cl_full = np.zeros(lmax_full + 1, dtype=float)
    Cl_full[fine_ells] = Cl_fine

    Wl_full = np.ones_like(Cl_full)
    if apply_pixwin:
        Wl = hp.pixwin(nside)
        Lw = min(len(Wl) - 1, lmax_full)
        Wl_full[:Lw + 1] = Wl[:Lw + 1]

    T = np.empty_like(ells_low, dtype=float)
    for b, (lo, hi) in enumerate(zip(ells_low, ells_high)):
        ell_bin = np.arange(int(lo), int(hi), dtype=int)
        ell_bin = ell_bin[(ell_bin >= 2) & (ell_bin <= lmax_full)]
        if ell_bin.size == 0:
            T[b] = 0.0
            continue
        w = (2.0 * ell_bin + 1.0)
        vals = Cl_full[ell_bin] * (Wl_full[ell_bin] ** 2)
        T[b] = float(np.sum(w * vals) / np.sum(w))
    return T


def _bin_edges_from_linear(nside: int, d_ell: int) -> np.ndarray:
    """Deterministic linear-bin edges for from_nside_linear; upper edge is exclusive."""
    lmax_full = 3 * nside - 1
    edges = np.arange(0, lmax_full + 1, d_ell, dtype=int)
    if edges[-1] != lmax_full + 1:
        edges = np.append(edges, lmax_full + 1)
    return edges

def _bin_meta_from_edges(edges: np.ndarray, nbins: int | None = None):
    """Build (ell_eff, ell_low, ell_high, nell) from edges; optionally trim to nbins."""
    if nbins is not None and nbins < len(edges) - 1:
        edges = edges[: nbins + 1]
    ells_low  = edges[:-1]
    ells_high = edges[1:]               # exclusive
    nell      = ells_high - ells_low
    ells_eff  = 0.5 * (ells_low + ells_high - 1)
    return ells_eff.astype(float), ells_low, ells_high, nell



# ------------------------------ autos measurement ----------------------------
def _auto_bandpowers(field: nmt.NmtField,
                     binning: nmt.NmtBin,
                     wsp_path: Path,
                     nside: int,
                     d_ell: int):
    """
    Returns (ell_eff, ell_low, ell_high, nell, C_b) with *matching* lengths.
    """
    wsp = nmt.NmtWorkspace()
    if wsp_path.exists():
        wsp.read_from(str(wsp_path))
    else:
        wsp.compute_coupling_matrix(field, field, binning)
        wsp.write_to(str(wsp_path))

    cl_coup = nmt.compute_coupled_cell(field, field)
    cl_dec  = wsp.decouple_cell(cl_coup)[0]              # length = N_bins used by NaMaster

    # Build edges deterministically, then trim to this exact number of bins
    edges = _bin_edges_from_linear(nside, d_ell)
    ell_eff, ell_lo, ell_hi, nell = _bin_meta_from_edges(edges, nbins=len(cl_dec))

    return ell_eff, ell_lo, ell_hi, nell, cl_dec





# ------------------------------ fitting b & N --------------------------------
def _fit_xN(ells: np.ndarray, Cdata: np.ndarray, T: np.ndarray,
            fsky2: float, nell: np.ndarray, use_weights: bool = True):
    """
    Solve for x and N in C = x*T + N by (optionally) weighted linear regression.
    Weights: Knox diagonal with (C^2)→(Cdata^2) proxy.
    """
    if use_weights:
        # diagonal Gaussian approx: Var(C_b) ~ 2 / ( (2l+1) f_sky^{(2)} Δl ) * (C_b)^2
        # use |data| as proxy; guard zeros
        sigma2 = 2.0 * (np.clip(Cdata, 1e-12, None)**2) / (np.maximum((2*ells+1)*fsky2*nell, 1e-12))
        w = 1.0 / np.clip(sigma2, 1e-30, None)
    else:
        w = np.ones_like(ells)

    S_TT = float(np.sum(w * T * T))
    S_T  = float(np.sum(w * T))
    S_11 = float(np.sum(w))
    S_dT = float(np.sum(w * Cdata * T))
    S_d  = float(np.sum(w * Cdata))
    Δ = S_TT*S_11 - S_T*S_T
    if Δ <= 0:
        return 0.0, 0.0, 0.0, 0.0

    x = ( S_11*S_dT - S_T*S_d ) / Δ
    N = ( S_TT*S_d  - S_T*S_dT ) / Δ

    # Parameter covariance = inverse Fisher (AᵀWA)⁻¹ = M/Δ, with the Knox weights
    # acting as inverse variances. Inflate by reduced χ² when the fit is poor so the
    # error reflects scatter. (The previous code used s2·M/Δ², an extra 1/Δ that
    # collapsed sig_x → 0.)
    yhat = x*T + N
    nb = max(1, ells.size - 2)
    red_chi2 = float(np.sum(w * (Cdata - yhat)**2) / nb)
    M = np.array([[S_11, -S_T], [-S_T, S_TT]], dtype=float)
    cov = M / Δ * max(1.0, red_chi2)
    sig_x = float(np.sqrt(max(cov[0, 0], 0.0)))
    return x, N, sig_x, red_chi2


# ------------------------------ plotting -------------------------------------
def _plot_bin(out_png: Path, ells, data, T, x, N):
    fig, ax = plt.subplots(1,1, figsize=(6.8,4.2))
    ax.plot(ells, data, "o", ms=4, label="data (C_b)")
    ax.plot(ells, x*T + N, "-", lw=1.6, label=f"fit: b^2={x:.3f}, N={N:.3e}")
    ax.axhline(0, lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{gg}$")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def _weighted_mean_z(z: np.ndarray, nz: np.ndarray) -> float:
    nz = np.clip(np.asarray(nz, float), 0, None)
    z  = np.asarray(z, float)
    if nz.sum() <= 0:  # fallback: simple midpoint
        return float(0.5 * (z.min() + z.max()))
    return float((nz * z).sum() / nz.sum())

def _plot_nz_all(nz_store, out_png: Path):
    """nz_store: dict[tracer] -> list of dicts {z, nz, label} (one per bin)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tracer, bins in nz_store.items():
        for ent in bins:
            z, nz = ent["z"], ent["nz"]
            ax.plot(z, nz, alpha=0.6, label=f"{tracer} {ent['label']}")
    ax.set_xlabel("z"); ax.set_ylabel("n(z) (arb. norm)")
    ax.set_title("DESI n(z) per bin")
    # Thin legend: only first entry per tracer to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    first_per_tracer = {}
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        t = l.split()[0]
        if t not in first_per_tracer:
            first_per_tracer[t] = True
            new_h.append(h); new_l.append(t)
    ax.legend(new_h, new_l, frameon=False, ncol=3)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160); plt.close(fig)

def _plot_bias_vs_z(bias_pts, out_png: Path):
    """bias_pts: dict[tracer] -> list of (z_mean, b, sigma_b)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tracer, rows in bias_pts.items():
        if not rows: continue
        zmean = np.array([r[0] for r in rows], float)
        b     = np.array([r[1] for r in rows], float)
        be    = np.array([r[2] for r in rows], float)
        ax.errorbar(zmean, b, yerr=be, fmt="o-", ms=4, capsize=3, label=tracer)
    ax.set_xlabel(r"$\bar z_{\rm bin}$"); ax.set_ylabel("linear bias b")
    ax.set_title("Bias vs redshift")
    ax.grid(ls=":", alpha=0.4); ax.legend(frameon=False)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160); plt.close(fig)



# ------------------------------ main routine ---------------------------------
def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    nside = int(args.nside)

    tracers = args.tracers or list(cfg["desi"]["bins"].keys())
    bins_cfg = cfg["desi"]["bins"]

    # Binning & pixwin
    binning = _make_bin(nside, args.ell_min, args.ell_max, args.delta_ell)
    ell_eff, ell_lo, ell_hi, nell_full = _bin_meta(binning, nside, args.delta_ell)
    sel = (ell_eff >= max(2, args.ell_min)) & (ell_eff <= args.ell_max)
    ell = ell_eff[sel]; nell = nell_full[sel]

    # cosmology
    cosmo = make_cosmology_from_config(cfg)
    mp    = MatterPowerSpectrum(cosmo)

    nz_store   = {}  # tracer -> list of {"z": zvec, "nz": nzvec, "label": f"z{ibin}"}
    bias_points = {} # tracer -> list of (zmean, b, sigma_b)


    priors = []
    plots_dir = Path(cfg["paths"]["spectra"]) / "desi" / "plots_bias"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for tracer in tracers:
        z_bins = bins_cfg[tracer]
        for ibin, (zmin, zmax) in enumerate(z_bins, start=1):
            # maps & mask
            mpath = _delta_map_path(cfg, tracer, nside, ibin)
            if not mpath.exists():
                LOGGER.warning("Missing map, skip: %s", mpath); continue
            dmap = hp.read_map(mpath, dtype=np.float64, verbose=False)
            mask_path = _mask_path(cfg, tracer, nside, args.use_joint_mask)
            msk  = hp.read_map(mask_path, dtype=np.float64, verbose=False)

            # field & fsky^2
            field, msk_apo, fsky2 = _build_field(dmap, msk, args.apod_deg)

            # measure autos
            wsp_dir = Path(cfg["paths"]["spectra"]) / "desi" / "workspaces"
            wsp_dir.mkdir(parents=True, exist_ok=True)
            wsp_path = wsp_dir / f"wsp_auto_{tracer}_z{ibin}_n{nside}_d{args.delta_ell}_apo{args.apod_deg:.2f}{'J' if args.use_joint_mask else 'L'}.fits"
            ell_full, ell_lo_full, ell_hi_full, nell_full, Cfull = _auto_bandpowers(
                field, binning, wsp_path, nside, args.delta_ell
            )
            sel   = (ell_full >= max(2, args.ell_min)) & (ell_full <= args.ell_max)
            ell   = ell_full[sel]
            nell  = nell_full[sel]
            Cdata = Cfull[sel]

            # theory template (unit bias), band-averaged
            nz_dir = Path(args.nz_dir) if args.nz_dir else None
            z, nz = _get_dndz_for_bin(cfg, tracer, ibin, nz_dir, float(zmin), float(zmax))
            T_full = _theory_unitbias_binned(
                cosmo, mp, z, nz, nside, ell_lo_full, ell_hi_full, args.apply_pixwin
            )
            T = T_full[sel]


            # fit x=b^2 and N over ell >= bias_lmin (skip systematics-dominated low-ell)
            fitsel = ell >= args.bias_lmin
            if fitsel.sum() < 3:
                LOGGER.warning("  <3 bandpowers above bias_lmin=%d; using all", args.bias_lmin)
                fitsel = np.ones_like(ell, dtype=bool)
            x, N, sig_x, _ = _fit_xN(ell[fitsel], Cdata[fitsel], T[fitsel],
                                     fsky2=fsky2, nell=nell[fitsel], use_weights=True)
            b = np.sqrt(max(x, 0.0))
            sigma_b = (0.5 * sig_x / max(b, 1e-12)) if b > 0 else 0.0

            pri = {
                "tracer": tracer, "ibin": ibin, "zmin": float(zmin), "zmax": float(zmax),
                "nside": nside, "ell_min": int(args.ell_min), "ell_max": int(args.ell_max),
                "delta_ell": int(args.delta_ell), "apod_deg": float(args.apod_deg),
                "use_joint_mask": bool(args.use_joint_mask),
                "b": float(b), "sigma_b": float(sigma_b),
                "N_shot_fit": float(N),
                "mask_path": str(mask_path),
                "delta_map": str(mpath),
            }
            priors.append(pri)
            LOGGER.info("Bias: %s z%d  b=%.3f ± %.3f  (N=%.3e, fsky2=%.3f)", tracer, ibin, b, sigma_b, N, fsky2)

            # store n(z) for summary plot
            nz_store.setdefault(tracer, []).append({"z": z, "nz": nz, "label": f"z{ibin}"})

            # mean z for the point (use weighted mean if nz provided)
            z_mean = _weighted_mean_z(z, nz)
            bias_points.setdefault(tracer, []).append((z_mean, b, sigma_b))


            if args.plot:
                _plot_bin(plots_dir / f"auto_{tracer}_z{ibin}.png", ell, Cdata, T, x, N)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.make_summary_plots:
        outdir = Path(args.plots_outdir)
        _plot_nz_all(nz_store, outdir / "desi_nz_per_bin.png")
        _plot_bias_vs_z(bias_points, outdir / "desi_bias_vs_z.png")



    with open(out, "w") as f:
        json.dump({"priors": priors}, f, indent=2)
    LOGGER.info("Wrote bias priors → %s", out)


if __name__ == "__main__":
    main()
