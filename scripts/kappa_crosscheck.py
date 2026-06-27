#!/usr/bin/env python3
"""
Cross-check DESI galaxy bias via DESI × Planck CMB lensing (κ) cross-correlation.

The CMB convergence κ cross-correlation with a galaxy sample provides an
independent bias constraint:

  C_ell^{gκ,data} ≈ b * C_ell^{gκ,theory}(b=1)

The GLS bias estimate per tomographic bin is:

  b_hat = (t^T C^{-1} d) / (t^T C^{-1} t),   σ_b = 1/√(t^T C^{-1} t)

where d = measured bandpowers, t = theory at b=1, and the Gaussian
bandpower covariance is:

  Var(Ĉ_b^{gκ}) = [C_b^{gg} C_b^{κκ} + (C_b^{gκ})²] / (f_sky Σ_ℓ (2ℓ+1))

Results are compared against the gg-based bias from calibrate_bias.py.

Inputs
------
  --kappa-map   Planck lensing κ FITS (e.g. COM_Lensing_4096_R3.00.fits)
  --kappa-mask  Planck lensing mask (auto-read from field 1 of kappa-map if absent)
  --nz-dir      Directory containing {TRACER}_NGCplusSGC_nz.txt files
  DESI delta maps and config as usual

Outputs
-------
  spectra/desi/gkappa_{TRACER}_z{ibin}_lmax{L}.npz
  results/kappa_bias_check.csv
  results/kappa_bias_check.png   (if matplotlib available)
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_crosscls import (
    _apodize,
    _build_fields,
    _compute_binned_cls,
    _load_map_ring,
    _load_mask_ring,
    _make_bin,
    _workspace_path,
)
from fit_isw_amplitudes import bandaverage, fit_gls, load_bias_priors

try:
    import pymaster as nmt
    HAS_NMT = True
except ImportError:
    HAS_NMT = False

try:
    from clank.cosmology import Cosmology
    from clank.matter_power import MatterPowerSpectrum
    from clank.number_counts_tracer import NumberCountsTracer
    from clank.cmb_lensing_tracer import CMBLensingTracer
    from clank.angular_cl import angular_cl as clank_angular_cl
    HAS_CLANK = True
except ImportError:
    HAS_CLANK = False

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

LOGGER = logging.getLogger("kappa_crosscheck")

NZ_FILE_PATTERN = "{tracer}_NGCplusSGC_nz.txt"


# ---------- CLI ----------

def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)

    # κ map inputs
    p.add_argument("--kappa-map", type=str, default=None,
                   help="Planck κ FITS file. Can also set paths.planck.kappa in config.")
    p.add_argument("--kappa-mask", type=str, default=None,
                   help="Lensing mask FITS. If absent, attempt to read field 1 from --kappa-map.")
    p.add_argument("--kappa-field", type=int, default=0,
                   help="FITS field index for κ map (default 0 = first column)")
    p.add_argument("--kappa-mask-field", type=int, default=1,
                   help="FITS field index for mask in κ file (default 1)")

    # n(z) inputs
    p.add_argument("--nz-dir", type=str, default=".",
                   help="Directory containing {TRACER}_NGCplusSGC_nz.txt files")

    # Survey/estimator
    p.add_argument("--tracers", nargs="+", default=None,
                   help="Tracers to process (default: all in config)")
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--delta-ell", type=int, default=10)
    p.add_argument("--apod-deg", type=float, default=1.0)
    p.add_argument("--apotype", choices=["C1", "C2"], default="C2")
    p.add_argument("--apodizer", choices=["healpy-gauss", "nmt", "none"], default="healpy-gauss")
    p.add_argument("--fsky", type=float, default=None)

    # Theory (for template computation)
    p.add_argument("--lmax-theory", type=int, default=200,
                   help="Max ell for theory C_ell^{gκ} computation")
    p.add_argument("--bias-priors", default="results/desi_bias_priors.json",
                   help="Calibrated bias priors from calibrate_bias.py (for comparison)")

    # Outputs
    p.add_argument("--outdir-spectra", type=str, default=None,
                   help="Output directory for gκ spectra NPZs (default: from config)")
    p.add_argument("--out-csv", default="results/kappa_bias_check.csv")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------- κ map loading ----------

def _read_planck_kappa(fits_path: str, field: int, nside_out: int):
    """
    Load a Planck lensing convergence map.

    Handles RING/NESTED ordering and re-grades to nside_out.
    Returns (kappa_map_ring_float64, nside_original).
    """
    try:
        m, hdr = hp.read_map(fits_path, field=field, dtype=np.float64,
                              h=True, verbose=False)
    except Exception:
        # Some Planck products use different HDU structures; try field=0
        m, hdr = hp.read_map(fits_path, field=0, dtype=np.float64,
                              h=True, verbose=False)

    nside_orig = hp.get_nside(m)
    ordering = "RING"
    for k, v in hdr:
        if k == "ORDERING":
            ordering = str(v).strip().upper()
            break

    if ordering == "NESTED":
        m = hp.reorder(m, n2r=True)

    if nside_orig != nside_out:
        LOGGER.info("Downgrading κ map: NSIDE %d → %d", nside_orig, nside_out)
        m = hp.ud_grade(m, nside_out=nside_out, order_in="RING", order_out="RING", power=-2)

    # Remove monopole and dipole on full sky before masking
    m = hp.remove_monopole(m, verbose=False)
    return m, nside_orig


def _read_kappa_mask(fits_path: str, mask_field: int, nside_out: int):
    """
    Read the Planck lensing analysis mask.
    Returns a float mask in [0,1] at nside_out, RING ordering.
    """
    try:
        msk, hdr = hp.read_map(fits_path, field=mask_field, dtype=np.float64,
                                h=True, verbose=False)
    except Exception as e:
        LOGGER.warning("Could not read mask field %d from %s: %s", mask_field, fits_path, e)
        return None

    ordering = "RING"
    for k, v in hdr:
        if k == "ORDERING":
            ordering = str(v).strip().upper()
            break
    if ordering == "NESTED":
        msk = hp.reorder(msk, n2r=True)

    if hp.get_nside(msk) != nside_out:
        msk = hp.ud_grade(msk, nside_out=nside_out, order_in="RING", order_out="RING", power=None)
    return np.clip(msk, 0.0, 1.0)


# ---------- n(z) loading ----------

def load_nz_bin(nz_dir: str, tracer: str, zmin: float, zmax: float):
    """
    Load n(z) for a tracer, trimmed to [zmin, zmax].
    Returns (z_mid, nz) on the bin grid, normalised to unit integral.
    """
    fname = Path(nz_dir) / NZ_FILE_PATTERN.format(tracer=tracer)
    if not fname.exists():
        raise FileNotFoundError(
            f"n(z) file not found: {fname}\n"
            f"  Expected format: zmid  zlow  zhigh  nz  Ntot  Vtot\n"
            f"  Pass --nz-dir pointing to the directory with these files.")

    data = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            zmid, zlo, zhi = float(cols[0]), float(cols[1]), float(cols[2])
            nz = float(cols[3])
            data.append((zmid, zlo, zhi, nz))

    data = np.array(data)  # (N, 4)
    zmid, zlo, zhi, nz = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Select rows whose bin centre falls in [zmin, zmax]
    sel = (zmid >= zmin) & (zmid <= zmax)
    if sel.sum() == 0:
        # Widen search slightly in case of edge effects
        sel = (zmid >= zmin - 0.01) & (zmid <= zmax + 0.01)
    if sel.sum() == 0:
        raise ValueError(f"No n(z) rows in z=[{zmin},{zmax}] for {tracer} in {fname}")

    z_out = zmid[sel]
    nz_out = np.clip(nz[sel], 0.0, None)

    # Normalise: unit integral (trapezoidal)
    norm = np.trapz(nz_out, z_out)
    if norm > 0:
        nz_out = nz_out / norm

    LOGGER.debug("n(z) for %s [%.2f,%.2f]: %d rows, z=[%.3f,%.3f]",
                 tracer, zmin, zmax, sel.sum(), z_out[0], z_out[-1])
    return z_out, nz_out


# ---------- theory templates ----------

def _build_cosmo_from_config(cfg: dict):
    """Instantiate a clank Cosmology from the theory.cosmo block in config."""
    c = cfg.get("theory", {}).get("cosmo", {})
    H0   = c.get("H0", 67.36)
    h    = H0 / 100.0
    ombh2 = c.get("ombh2", 0.02237)
    omch2 = c.get("omch2", 0.1200)
    Omega_b = ombh2 / h**2
    Omega_c = omch2 / h**2
    return Cosmology(
        h=h, Omega_c=Omega_c, Omega_b=Omega_b,
        n_s=c.get("ns", 0.9649),
        tau=c.get("tau", 0.054),
        As=c.get("As", 2.1e-9),
        w0=c.get("w0", -1.0),
        wa=c.get("wa", 0.0),
        zmax=6.0, nz=300, kmax=10.0, nk=256,
        nonlinear=False,
    )


def compute_theory_templates(cfg: dict, tracer: str, z_nz: np.ndarray, nz: np.ndarray,
                              lmax: int):
    """
    Compute C_ell^{gκ}(b=1), C_ell^{gg}(b=1), and C_ell^{κκ} using clank.

    Returns (ell, cl_gkappa, cl_gg, cl_kk) all at unit bias.
    """
    if not HAS_CLANK:
        raise RuntimeError(
            "clank not available. Install it or provide --kappa-template.\n"
            "pip install -e /path/to/clank")

    cosmo = _build_cosmo_from_config(cfg)
    mp    = MatterPowerSpectrum(cosmo, nonlinear=False)
    kappa = CMBLensingTracer(cosmo)

    # Unit bias for the template — the GLS fit recovers the actual b
    b_unit = np.ones_like(z_nz)
    gal = NumberCountsTracer(
        cosmo,
        dndz=(z_nz, nz),
        bias=(z_nz, b_unit),
        name=tracer,
    )

    ell = np.arange(0, lmax + 1, dtype=int)
    LOGGER.info("Computing theory C_ell^{gκ} for %s (lmax=%d)...", tracer, lmax)
    cl_gk  = clank_angular_cl(cosmo, gal,   kappa, ell, mode="full_z", mp=mp)
    cl_gg  = clank_angular_cl(cosmo, gal,   gal,   ell, mode="full_z", mp=mp)
    cl_kk  = clank_angular_cl(cosmo, kappa, kappa, ell, mode="full_z", mp=mp)

    return ell, cl_gk, cl_gg, cl_kk


# ---------- Gaussian covariance ----------

def gkappa_gaussian_cov(cl_gk_th, cl_gg_th, cl_kk_th, ell_lo, ell_hi, fsky):
    """
    Diagonal Gaussian bandpower covariance for C_ell^{gκ}:

      Var(Ĉ_b^{gκ}) = [C_b^{gg} C_b^{κκ} + (C_b^{gκ})²] / (f_sky Σ_ℓ∈b (2ℓ+1))
    """
    var = np.zeros(len(cl_gk_th))
    for b in range(len(var)):
        ells = np.arange(int(ell_lo[b]), int(ell_hi[b]) + 1)
        n_modes = fsky * np.sum(2 * ells + 1)
        var[b] = (cl_gg_th[b] * cl_kk_th[b] + cl_gk_th[b]**2) / n_modes
    return np.diag(var)


# ---------- cross-spectrum computation ----------

def compute_gkappa_spectrum(delta_map_p: Path, kappa_map: np.ndarray, mask: np.ndarray,
                             nside: int, ell_min: int, ell_max: int, delta_ell: int,
                             apod_deg: float, apotype: str, apod_method: str,
                             wsp_cache_root: Path, tracer: str, ibin: int):
    """
    Cross-correlate a DESI delta map with the prepared κ map using NaMaster.

    κ map is already in memory (float array at nside); delta map is read from disk.
    Returns (ell_eff, ell_lo, ell_hi, cl_dec, fsky_apod).
    """
    if not HAS_NMT:
        raise RuntimeError("pymaster required: pip install pymaster")

    for p in (delta_map_p,):
        if not p.exists():
            raise FileNotFoundError(f"DESI delta map not found: {p}")

    # Load galaxy overdensity map
    m_delta = _load_map_ring(delta_map_p, nside)

    # Flag bad pixels in both maps
    finite_d = np.isfinite(m_delta) & (m_delta != hp.UNSEEN)
    finite_k = np.isfinite(kappa_map) & (kappa_map != hp.UNSEEN)

    msk_raw = mask * finite_d.astype(float) * finite_k.astype(float)
    fsky_raw = float(np.mean(msk_raw > 0))
    if fsky_raw <= 0:
        raise RuntimeError(f"Joint mask empty for {tracer} z{ibin}")
    LOGGER.info("  f_sky (raw) = %.4f  [%s z%d]", fsky_raw, tracer, ibin)

    msk_apo = _apodize(msk_raw, apod_deg, apotype, method=apod_method)
    fsky_apo = float(np.mean(msk_apo > 0))
    LOGGER.info("  f_sky (apod) = %.4f", fsky_apo)

    # NaMaster fields
    f_g = nmt.NmtField(msk_apo, [m_delta])
    f_k = nmt.NmtField(msk_apo, [kappa_map])

    binning = _make_bin(nside, ell_min, ell_max, delta_ell)
    wsp_path = _workspace_path(wsp_cache_root, nside, delta_ell, apod_deg, apotype)
    wsp_path = wsp_path.with_name(wsp_path.stem + f"_gkappa_{tracer}z{ibin}" + wsp_path.suffix)

    ell, elo, ehi, cl, _ = _compute_binned_cls(
        f_g, f_k, binning, wsp_path, lmin_sel=ell_min, lmax_sel=ell_max)
    return ell, elo, ehi, cl, fsky_apo


# ---------- plotting ----------

def plot_bias_comparison(rows: list, bias_map: dict, out_path: Path):
    """Plot b_kappa vs b_gg per bin/tracer for quick visual comparison."""
    if not HAS_MPL:
        return

    tracers_seen = sorted({r["tracer"] for r in rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    col_map = {t: colors[i % len(colors)] for i, t in enumerate(tracers_seen)}

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(rows))
    labels = [f"{r['tracer']} z{r['ibin']}" for r in rows]

    b_kappa = np.array([r["b_kappa"] for r in rows])
    sb_kappa = np.array([r["sigma_b_kappa"] for r in rows])

    ax.errorbar(xs, b_kappa, yerr=sb_kappa, fmt="o", label="b (gκ)", ms=6, capsize=3)

    # Overlay gg-based bias where available
    b_gg = []
    for r in rows:
        prior = bias_map.get((r["tracer"], r["ibin"]), {})
        b_gg.append(prior.get("b", None))

    valid = [(i, b) for i, b in enumerate(b_gg) if b is not None and b != 0]
    if valid:
        xi, bi = zip(*valid)
        ax.scatter(xi, bi, marker="s", s=40, zorder=5, label="b (gg auto)")

    ax.axhline(1.0, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Galaxy bias  b")
    ax.set_title("Bias cross-check: gκ vs gg auto")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved bias comparison plot → %s", out_path)


def plot_gkappa_spectrum(ell, cl, ell_lo, ell_hi, cl_gk_theory, tracer, ibin, out_path):
    """Plot measured gκ bandpowers vs theory template."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ell, cl, fmt="o", ms=4, capsize=2, label="data")

    ell_th = np.arange(len(cl_gk_theory))
    ax.plot(ell_th[2:], cl_gk_theory[2:], lw=1.5, ls="--", alpha=0.8, label="theory (b=1)")
    ax.axhline(0, ls=":", lw=0.8, color="gray")

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{g\kappa}$")
    ax.set_title(f"{tracer} z{ibin}  ×  CMB κ")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved gκ spectrum plot → %s", out_path)


# ---------- main ----------

def main():
    args = parse_args()
    setup_logging(args.verbose)

    if not HAS_NMT:
        LOGGER.error("pymaster is required: pip install pymaster")
        sys.exit(1)

    cfg = load_config(args.config)
    tracers = args.tracers or list(cfg["desi"]["bins"].keys())

    # Output paths
    spectra_root = (Path(args.outdir_spectra) if args.outdir_spectra
                    else Path(cfg["paths"]["spectra"]) / "desi")
    spectra_root.mkdir(parents=True, exist_ok=True)
    wsp_cache = spectra_root / "workspaces"
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve κ map path: CLI > config
    kappa_fits = args.kappa_map or cfg.get("paths", {}).get("planck", {}).get("kappa")
    if not kappa_fits:
        LOGGER.error("No κ map provided. Use --kappa-map or set paths.planck.kappa in config.")
        sys.exit(1)
    kappa_fits = str(kappa_fits)

    # --- Load and prepare κ map once ---
    LOGGER.info("Loading κ map from %s (field=%d)...", kappa_fits, args.kappa_field)
    kappa_map, nside_orig = _read_planck_kappa(kappa_fits, args.kappa_field, args.nside)
    LOGGER.info("κ map: NSIDE %d → %d,  rms=%.4g", nside_orig, args.nside, float(np.std(kappa_map[kappa_map != hp.UNSEEN])))

    # --- Load κ mask ---
    kappa_mask = None
    if args.kappa_mask:
        LOGGER.info("Loading κ mask from %s", args.kappa_mask)
        kappa_mask = _read_kappa_mask(args.kappa_mask, 0, args.nside)
    else:
        LOGGER.info("Attempting to read κ mask from field %d of κ file...", args.kappa_mask_field)
        kappa_mask = _read_kappa_mask(kappa_fits, args.kappa_mask_field, args.nside)
        if kappa_mask is None:
            LOGGER.warning("Could not read κ mask; proceeding with full κ footprint.")
            kappa_mask = (kappa_map != hp.UNSEEN).astype(float)

    # --- Load bias priors for comparison ---
    bias_map = load_bias_priors(args.bias_priors)

    rows = []

    for tracer in tracers:
        bins = cfg["desi"]["bins"].get(tracer, [])
        if not bins:
            LOGGER.warning("No bins configured for %s; skipping", tracer)
            continue

        # Load n(z) once per tracer
        try:
            nz_arr = np.loadtxt(
                Path(args.nz_dir) / NZ_FILE_PATTERN.format(tracer=tracer),
                comments="#")
        except FileNotFoundError as e:
            LOGGER.error("n(z) file missing for %s: %s", tracer, e)
            continue
        zmid_all, zlo_all, zhi_all, nz_all = (nz_arr[:, 0], nz_arr[:, 1],
                                               nz_arr[:, 2], nz_arr[:, 3])

        for ibin, (zmin, zmax) in enumerate(bins, start=1):
            LOGGER.info("── %s z%d [%.2f, %.2f] ──", tracer, ibin, zmin, zmax)

            out_npz = spectra_root / f"gkappa_{tracer}_z{ibin}_lmax{args.ell_max}.npz"
            if out_npz.exists() and not args.overwrite:
                LOGGER.info("Loading existing spectrum %s", out_npz.name)
                spec = np.load(out_npz, allow_pickle=True)
                ell_eff = spec["ell"]
                elo, ehi = spec["ell_edges_lo"], spec["ell_edges_hi"]
                cl_data = spec["cl"]
                fsky_val = float(spec.get("fsky", args.fsky or 0.3))
            else:
                # Build joint mask: DESI LSS mask ∩ κ mask
                masks_root = Path(cfg["paths"]["masks"])
                lss_mask_p = masks_root / f"desi_lssmask_{tracer}_nside{args.nside}.fits.gz"
                joint_mask_p = masks_root / f"planck_desi_joint_nside{args.nside}.fits.gz"

                if lss_mask_p.exists():
                    lss_mask = _load_mask_ring(str(lss_mask_p), args.nside)
                elif joint_mask_p.exists():
                    lss_mask = _load_mask_ring(str(joint_mask_p), args.nside)
                    LOGGER.info("Using joint mask (no per-tracer LSS mask found)")
                else:
                    LOGGER.error("No mask found for %s; skipping", tracer)
                    continue

                combined_mask = lss_mask * kappa_mask

                maps_root = Path(cfg["paths"]["maps"])
                delta_p = maps_root / "desi" / tracer / f"nside{args.nside}" / f"delta_{tracer}_z{ibin}.fits.gz"

                try:
                    ell_eff, elo, ehi, cl_data, fsky_val = compute_gkappa_spectrum(
                        delta_p, kappa_map, combined_mask,
                        args.nside, args.ell_min, args.ell_max, args.delta_ell,
                        args.apod_deg, args.apotype, args.apodizer,
                        wsp_cache, tracer, ibin,
                    )
                except FileNotFoundError as e:
                    LOGGER.error("Missing input: %s", e)
                    continue
                except Exception as e:
                    LOGGER.error("NaMaster failed for %s z%d: %s", tracer, ibin, e)
                    continue

                np.savez(out_npz, ell=ell_eff, cl=cl_data,
                         ell_edges_lo=elo, ell_edges_hi=ehi,
                         tracer=tracer, ibin=ibin, nside=args.nside, fsky=fsky_val,
                         zmin=zmin, zmax=zmax)
                LOGGER.info("Saved gκ spectrum → %s", out_npz)

            # --- Theory template ---
            # Get per-bin n(z)
            sel = (zmid_all >= zmin - 0.005) & (zmid_all <= zmax + 0.005)
            z_bin = zmid_all[sel]
            nz_bin = np.clip(nz_all[sel], 0.0, None)
            norm = np.trapz(nz_bin, z_bin) if len(z_bin) > 1 else 1.0
            if norm > 0:
                nz_bin = nz_bin / norm

            if len(z_bin) < 2:
                LOGGER.error("Too few n(z) points for %s z%d [%.2f,%.2f]; skipping fit",
                             tracer, ibin, zmin, zmax)
                continue

            try:
                ell_th, cl_gk_th, cl_gg_th, cl_kk_th = compute_theory_templates(
                    cfg, tracer, z_bin, nz_bin, args.lmax_theory)
            except Exception as e:
                LOGGER.error("Theory computation failed for %s z%d: %s", tracer, ibin, e)
                if not HAS_CLANK:
                    LOGGER.error("Install clank: pip install -e /path/to/clank")
                continue

            # Band-average theory to match data ell grid
            cl_gk_band = bandaverage(ell_th, cl_gk_th, elo, ehi)
            cl_gg_band = bandaverage(ell_th, cl_gg_th, elo, ehi)
            cl_kk_band = bandaverage(ell_th, cl_kk_th, elo, ehi)

            # Effective fsky for covariance
            fsky = args.fsky or fsky_val

            cov = gkappa_gaussian_cov(cl_gk_band, cl_gg_band, cl_kk_band, elo, ehi, fsky)

            # GLS bias fit:  d ≈ b * t  →  b_hat = (t C⁻¹ d)/(t C⁻¹ t)
            b_hat, sigma_b = fit_gls(cl_data, cov, cl_gk_band)
            LOGGER.info("  b_kappa = %.3f ± %.3f", b_hat, sigma_b)

            # Comparison with gg-based bias
            prior = bias_map.get((tracer, ibin), {})
            b_gg = prior.get("b", None)
            sigma_b_gg = prior.get("sigma_b", None)
            if b_gg:
                pull = (b_hat - b_gg) / np.sqrt(sigma_b**2 + (sigma_b_gg or 0)**2)
                LOGGER.info("  b_gg  = %.3f ± %.3f  (pull = %.2f σ)",
                            b_gg, sigma_b_gg or 0, pull)
            else:
                pull = float("nan")

            rows.append({
                "tracer": tracer, "ibin": ibin, "zmin": zmin, "zmax": zmax,
                "b_kappa": round(b_hat, 5), "sigma_b_kappa": round(sigma_b, 5),
                "b_gg": b_gg or "", "sigma_b_gg": sigma_b_gg or "",
                "pull_sigma": round(pull, 3) if np.isfinite(pull) else "",
                "lmin": int(elo[0]), "lmax": int(ehi[-1]), "fsky": round(fsky, 4),
            })

            # Per-bin spectrum plot
            if HAS_MPL:
                plot_dir = results_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_gkappa_spectrum(
                    ell_eff, cl_data, elo, ehi, cl_gk_th,
                    tracer, ibin,
                    plot_dir / f"gkappa_{tracer}_z{ibin}.png")

    # --- Write summary CSV ---
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tracer", "ibin", "zmin", "zmax",
                  "b_kappa", "sigma_b_kappa", "b_gg", "sigma_b_gg",
                  "pull_sigma", "lmin", "lmax", "fsky"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        LOGGER.info("Wrote %d rows → %s", len(rows), out_csv)
        # Print table to stdout
        print(f"\n{'Tracer':>6} {'bin':>4}  {'z range':>12}  {'b_κ':>8}  {'σb_κ':>8}  {'b_gg':>8}  {'pull':>6}")
        print("-" * 64)
        for r in rows:
            zr  = f"[{r['zmin']:.2f},{r['zmax']:.2f}]"
            bgg = f"{r['b_gg']:8.3f}" if r["b_gg"] != "" else f"{'—':>8}"
            pul = f"{r['pull_sigma']:6.2f}" if r["pull_sigma"] != "" else f"{'—':>6}"
            print(f"{r['tracer']:>6} {r['ibin']:>4}  {zr:>12}  "
                  f"{r['b_kappa']:8.3f}  {r['sigma_b_kappa']:8.3f}  {bgg}  {pul}")

        # Summary comparison plot
        if HAS_MPL:
            plot_bias_comparison(rows, bias_map,
                                 results_dir / "kappa_bias_check.png")
    else:
        LOGGER.warning("No bins fitted — check inputs and logs.")


if __name__ == "__main__":
    main()
