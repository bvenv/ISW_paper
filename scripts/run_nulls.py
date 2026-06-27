#!/usr/bin/env python3
"""
Run null/systematics suites for DESI×Planck ISW analysis.

CMB suite  (--suite cmb)
  curl        — cross δ × R90(T): T map rotated 90° on the sky; breaks physical
                gT correlation while preserving CMB statistics
  comp_diff   — cross δ × (T_SMICA − T_Commander)/√2: component-separation null;
                ISW cancels, correlated foreground residuals would not
  ell_range   — re-fit A_ISW on ℓ sub-ranges using existing main spectra;
                checks scale-dependence without new map computation

LSS suite  (--suite lss)
  ngc_sgc     — cross T × δ_NGC and T × δ_SGC separately; tests sky-cap consistency
                (requires per-cap delta maps built with desi_build_maps.py --cap NGC/SGC)
  random_null — cross T × δ_randoms: overdensity from random catalog alone should
                give zero signal (tests systematic contamination of randoms)

Outputs
-------
  {spectra_root}/nulls/{suite}/{test}/gT_{TRACER}_z{ibin}.npz
  {spectra_root}/nulls/{suite}/null_table.csv
    columns: tracer, ibin, test, A, sigmaA, chi2, ndof, pvalue
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import yaml

# NaMaster utilities shared with compute_crosscls.py
from compute_crosscls import (
    _apodize,
    _build_fields,
    _compute_binned_cls,
    _load_map_ring,
    _load_mask_ring,
    _make_bin,
    _workspace_path,
)
from fit_isw_amplitudes import (
    bandaverage,
    fit_gls,
    fsky_from_mask,
    gaussian_cov,
    load_bias_priors,
    load_theory_templates,
)

try:
    import pymaster as nmt
except ImportError as e:
    raise RuntimeError("pymaster (NaMaster) required: pip install pymaster") from e

try:
    from scipy.stats import chi2 as chi2_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

LOGGER = logging.getLogger("run_nulls")


# ---------- CLI ----------

def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--suite", choices=["cmb", "lss"], required=True)
    p.add_argument("--tests", nargs="+", default=None,
                   help="Subset of tests to run (default: all for the suite). "
                        "CMB: curl, comp_diff, ell_range. LSS: ngc_sgc, random_null.")
    p.add_argument("--tracers", nargs="+", default=None,
                   help="Tracers to process (default: all in config)")
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--delta-ell", type=int, default=10)
    p.add_argument("--apod-deg", type=float, default=1.0)
    p.add_argument("--apotype", choices=["C1", "C2"], default="C2")
    p.add_argument("--apodizer", choices=["healpy-gauss", "nmt", "none"], default="healpy-gauss")
    p.add_argument("--cmb-label", default="SMICA")
    p.add_argument("--templates-dir", default="templates/isw")
    p.add_argument("--bias-priors", default="results/desi_bias_priors.json")
    p.add_argument("--fsky", type=float, default=None)
    p.add_argument("--sim-cov-file", default=None,
                   help="results/sim_cov_{TRACER}.npz from sim_covariance.py. When given, the "
                        "curl and ell_range nulls use the (Hartlap-corrected) sims covariance "
                        "instead of the optimistic analytic Gaussian one.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------- path helpers ----------

def _maps_root(cfg): return Path(cfg["paths"]["maps"])
def _masks_root(cfg): return Path(cfg["paths"]["masks"])
def _spectra_root(cfg): return Path(cfg["paths"]["spectra"]) / "desi"

def delta_map_path(cfg, tracer, ibin, nside, cap="ANY"):
    tag = f"_cap{cap}" if cap not in ("ANY", "") else ""
    return _maps_root(cfg) / "desi" / tracer / f"nside{nside}" / f"delta_{tracer}{tag}_z{ibin}.fits.gz"

def cmb_map_path(cfg, label, nside):
    return _maps_root(cfg) / "planck" / f"nside{nside}" / f"cmb_T_{label}.fits.gz"

def joint_mask_path(cfg, nside):
    return _masks_root(cfg) / f"planck_desi_joint_nside{nside}.fits.gz"

def lss_mask_path(cfg, tracer, nside):
    return _masks_root(cfg) / f"desi_lssmask_{tracer}_nside{nside}.fits.gz"


# ---------- core cross-spectrum computation ----------

def _cross_spectrum(delta_map_p, cmb_map_p, mask_p, nside, ell_min, ell_max,
                    delta_ell, apod_deg, apotype, apod_method, wsp_cache_root,
                    wsp_tag_extra=""):
    """
    Compute gT bandpowers for a given pair of (galaxy delta, CMB T) maps.
    Returns (ell_eff, ell_lo, ell_hi, cl_dec, fsky).
    """
    for path in (delta_map_p, cmb_map_p, mask_p):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing input: {path}")

    joint_mask = _load_mask_ring(mask_p, nside)
    f_g, f_T, msk_apo, fsky = _build_fields(
        delta_map_p, cmb_map_p, joint_mask, apod_deg, apotype, apod_method)

    binning = _make_bin(nside, ell_min, ell_max, delta_ell)

    wsp_path = _workspace_path(wsp_cache_root, nside, delta_ell, apod_deg, apotype)
    # Give null workspaces a unique tag so they don't collide with main analysis
    if wsp_tag_extra:
        wsp_path = wsp_path.with_name(wsp_path.stem + wsp_tag_extra + wsp_path.suffix)

    ell, elo, ehi, cl, _ = _compute_binned_cls(
        f_g, f_T, binning, wsp_path, lmin_sel=ell_min, lmax_sel=ell_max)
    return ell, elo, ehi, cl, fsky


def _save_null_npz(out_path, ell, elo, ehi, cl, tracer, ibin, test_name, nside, fsky, **extra):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, ell=ell, cl=cl, ell_edges_lo=elo, ell_edges_hi=ehi,
             tracer=tracer, ibin=ibin, null_test=test_name, nside=nside, fsky=fsky,
             **extra)
    LOGGER.info("Wrote null spectrum → %s", out_path)


# ---------- chi² and pvalue helper ----------

def _chi2_null(cl_data, cov, cinv=None):
    """chi² = d^T C^{-1} d (null hypothesis: cl_data = 0). cinv overrides inv(cov)."""
    if cinv is None:
        cinv = np.linalg.inv(cov)
    chi2 = float(cl_data @ cinv @ cl_data)
    ndof = len(cl_data)
    pvalue = float(chi2_dist.sf(chi2, ndof)) if HAS_SCIPY else float("nan")
    return chi2, ndof, pvalue


def _gls_A(cl_data, cl_theory, cov, cinv=None):
    """GLS amplitude and error, optionally with a precomputed inverse cinv."""
    if cinv is None:
        cinv = np.linalg.inv(cov)
    tCt = float(cl_theory @ cinv @ cl_theory)
    return float((cl_theory @ cinv @ cl_data) / tCt), float(1.0 / np.sqrt(tCt))


def _fit_null(cl_data, cov, cl_theory, cinv=None):
    """Fit A and compute chi² of residual. cinv (e.g. Hartlap·inv(sim_cov)) overrides inv(cov)."""
    A, sigma_A = _gls_A(cl_data, cl_theory, cov, cinv)
    residual = cl_data - A * cl_theory
    chi2, ndof, pvalue = _chi2_null(residual, cov, cinv)
    return A, sigma_A, chi2, ndof, pvalue


def _sim_cov_cinv(sim_cov_file, ibin, ell_lo, ell_hi):
    """
    Hartlap-corrected inverse of the sims-covariance sub-block whose bands match
    (ell_lo, ell_hi) for tomographic bin `ibin`. Returns None (→ caller uses the
    analytic cov) if the file is absent or any band can't be matched.
    """
    if not sim_cov_file:
        return None
    p = Path(sim_cov_file)
    if not p.exists():
        LOGGER.warning("sim-cov file %s not found; using analytic cov", sim_cov_file)
        return None
    d = np.load(p)
    nband, nsims = int(d["nband"]), int(d["nsims"])
    blk = d["cov"][(ibin - 1) * nband: ibin * nband, (ibin - 1) * nband: ibin * nband]
    slo, shi = d["ell_lo"].astype(int), d["ell_hi"].astype(int)
    idx = []
    for lo, hi in zip(np.asarray(ell_lo, int), np.asarray(ell_hi, int)):
        m = np.where((slo == lo) & (shi == hi))[0]
        if len(m) == 0:
            LOGGER.warning("sim-cov band (%d,%d) not found; using analytic cov", lo, hi)
            return None
        idx.append(int(m[0]))
    sub = blk[np.ix_(idx, idx)]
    hartlap = (nsims - len(idx) - 2) / (nsims - 1)
    return hartlap * np.linalg.inv(sub)


def _theory_for_bin(templates_dir, tracer, ibin, ell_lo, ell_hi, fsky, cl_gg_scale=1.0):
    """Load templates and return (cl_gT_band, cov) for GLS fitting."""
    ell_th, cl_gT_th, cl_gg_th, cl_TT_th = load_theory_templates(templates_dir, tracer, ibin)
    cl_gT_b = bandaverage(ell_th, cl_gT_th, ell_lo, ell_hi)
    cl_gg_b = bandaverage(ell_th, cl_gg_th * cl_gg_scale, ell_lo, ell_hi)
    cl_TT_b = bandaverage(ell_th, cl_TT_th, ell_lo, ell_hi)
    cov = gaussian_cov(cl_gT_b, cl_gg_b, cl_TT_b, ell_lo, ell_hi, fsky)
    return cl_gT_b, cov


# ---------- CMB null tests ----------

def _rotate_cmb_90(cmb_map_p, nside):
    """Return T map rotated 90° in longitude — breaks gT correlation, keeps CMB stats."""
    T = _load_map_ring(cmb_map_p, nside)
    alm = hp.map2alm(T, lmax=3 * nside - 1)
    rot = hp.Rotator(rot=[90.0, 0.0, 0.0], deg=True)
    alm_rot = rot.rotate_alm(alm)
    return hp.alm2map(alm_rot, nside, verbose=False)


def run_curl(cfg, tracer, ibin, nside, ell_min, ell_max, delta_ell,
             apod_deg, apotype, apod_method, cmb_label, null_root,
             templates_dir, fsky_override, wsp_cache_root, overwrite, sim_cov_file=None):
    """Cross δ with T rotated by 90°. Expectation: A_ISW ≈ 0."""
    out = null_root / "curl" / f"gT_{tracer}_z{ibin}.npz"
    if out.exists() and not overwrite:
        LOGGER.info("Skip existing %s", out)
        return _load_null_result(out, templates_dir, tracer, ibin, fsky_override, sim_cov_file)

    LOGGER.info("[curl] %s z%d — rotating T map by 90°", tracer, ibin)
    cmb_p = cmb_map_path(cfg, cmb_label, nside)
    T_rot = _rotate_cmb_90(cmb_p, nside)

    # Write rotated map to a temp location
    tmp_cmb = null_root / "curl" / f"_tmp_cmb_rot90_nside{nside}.fits.gz"
    tmp_cmb.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(tmp_cmb), T_rot, overwrite=True)

    delta_p = delta_map_path(cfg, tracer, ibin, nside)
    mask_p = joint_mask_path(cfg, nside)

    ell, elo, ehi, cl, fsky = _cross_spectrum(
        delta_p, tmp_cmb, mask_p, nside, ell_min, ell_max,
        delta_ell, apod_deg, apotype, apod_method, wsp_cache_root,
        wsp_tag_extra="_curl")
    tmp_cmb.unlink(missing_ok=True)

    _save_null_npz(out, ell, elo, ehi, cl, tracer, ibin, "curl", nside, fsky,
                   cmb_label=cmb_label)

    fsky = fsky_override or fsky
    cl_gT_b, cov = _theory_for_bin(templates_dir, tracer, ibin, elo, ehi, fsky)
    cinv = _sim_cov_cinv(sim_cov_file, ibin, elo, ehi)
    A, sA, chi2, ndof, pval = _fit_null(cl, cov, cl_gT_b, cinv=cinv)
    LOGGER.info("  curl A=%.3f±%.3f  chi2/dof=%.2f/%d  p=%.3f%s", A, sA, chi2, ndof, pval,
                "  [sim-cov]" if cinv is not None else "")
    return {"test": "curl", "tracer": tracer, "ibin": ibin, "A": A, "sigmaA": sA,
            "chi2": chi2, "ndof": ndof, "pvalue": pval}


def run_comp_diff(cfg, tracer, ibin, nside, ell_min, ell_max, delta_ell,
                  apod_deg, apotype, apod_method, null_root,
                  templates_dir, fsky_override, wsp_cache_root, overwrite,
                  label_a="SMICA", label_b="Commander"):
    """
    Cross δ with (T_A − T_B)/√2. Both maps contain the same ISW signal, so
    it cancels; surviving signal is correlated foreground residuals or noise.
    """
    out = null_root / "comp_diff" / f"gT_{tracer}_z{ibin}.npz"
    if out.exists() and not overwrite:
        LOGGER.info("Skip existing %s", out)
        return _load_null_result(out, templates_dir, tracer, ibin, fsky_override)

    cmb_a = cmb_map_path(cfg, label_a, nside)
    cmb_b = cmb_map_path(cfg, label_b, nside)
    for p in (cmb_a, cmb_b):
        if not p.exists():
            LOGGER.warning("[comp_diff] Missing prepared map %s; skipping this null", p)
            return None

    LOGGER.info("[comp_diff] %s z%d — (%s − %s)/√2", tracer, ibin, label_a, label_b)
    T_a = _load_map_ring(cmb_a, nside)
    T_b = _load_map_ring(cmb_b, nside)
    T_diff = (T_a - T_b) / np.sqrt(2.0)

    tmp_cmb = null_root / "comp_diff" / f"_tmp_diff_{label_a}_{label_b}_nside{nside}.fits.gz"
    tmp_cmb.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(tmp_cmb), T_diff, overwrite=True)

    delta_p = delta_map_path(cfg, tracer, ibin, nside)
    mask_p = joint_mask_path(cfg, nside)

    ell, elo, ehi, cl, fsky = _cross_spectrum(
        delta_p, tmp_cmb, mask_p, nside, ell_min, ell_max,
        delta_ell, apod_deg, apotype, apod_method, wsp_cache_root,
        wsp_tag_extra="_compdiff")
    tmp_cmb.unlink(missing_ok=True)

    _save_null_npz(out, ell, elo, ehi, cl, tracer, ibin, "comp_diff", nside, fsky,
                   label_a=label_a, label_b=label_b)

    # Theory covariance for the difference: same gg but TT → TT_diff ≈ 0 (noise-dominated)
    # Use half the original TT as a conservative estimate
    fsky = fsky_override or fsky
    cl_gT_b, cov = _theory_for_bin(templates_dir, tracer, ibin, elo, ehi, fsky)
    # For the null, we test against zero: chi² = d^T C^{-1} d
    chi2, ndof, pval = _chi2_null(cl, cov)
    LOGGER.info("  comp_diff chi2/dof=%.2f/%d  p=%.3f", chi2, ndof, pval)
    return {"test": "comp_diff", "tracer": tracer, "ibin": ibin, "A": float("nan"),
            "sigmaA": float("nan"), "chi2": chi2, "ndof": ndof, "pvalue": pval}


def run_ell_range(cfg, tracer, ibin, ell_max, null_root, templates_dir,
                  fsky_override, bias_map, overwrite, sim_cov_file=None):
    """
    Re-fit A_ISW on ℓ sub-ranges from the existing main-analysis spectrum.
    No new map computation — purely a post-processing check.
    """
    spectra_root = _spectra_root(cfg)
    main_npz = spectra_root / f"gT_{tracer}_z{ibin}_lmax{ell_max}.npz"
    if not main_npz.exists():
        LOGGER.warning("[ell_range] Main spectrum not found: %s; skipping", main_npz)
        return []

    spec = np.load(main_npz, allow_pickle=True)
    cl_data = spec["cl"]
    ell_eff = spec["ell"]
    ell_lo  = spec["ell_edges_lo"]
    ell_hi  = spec["ell_edges_hi"]

    prior = bias_map.get((tracer, ibin), {})
    fsky = fsky_override
    if fsky is None and "mask_path" in spec:
        fsky = fsky_from_mask(str(spec["mask_path"]))
    if fsky is None:
        fsky = 0.3

    ell_th, cl_gT_th, cl_gg_th, cl_TT_th = load_theory_templates(templates_dir, tracer, ibin)

    # Define sub-range splits within the available band
    l0, l1 = int(ell_lo[0]), int(ell_hi[-1])
    mid = (l0 + l1) // 2
    sub_ranges = [
        ("low_ell",  l0, mid),
        ("high_ell", mid + 1, l1),
    ]

    rows = []
    for tag, lmin_sub, lmax_sub in sub_ranges:
        sel = (ell_eff >= lmin_sub) & (ell_eff <= lmax_sub)
        if sel.sum() < 2:
            LOGGER.debug("[ell_range] %s z%d %s: too few bands; skipping", tracer, ibin, tag)
            continue

        cl_sub = cl_data[sel]
        elo_sub = ell_lo[sel]
        ehi_sub = ell_hi[sel]

        cl_gT_b = bandaverage(ell_th, cl_gT_th, elo_sub, ehi_sub)
        cl_gg_b = bandaverage(ell_th, cl_gg_th, elo_sub, ehi_sub)
        cl_TT_b = bandaverage(ell_th, cl_TT_th, elo_sub, ehi_sub)
        cov = gaussian_cov(cl_gT_b, cl_gg_b, cl_TT_b, elo_sub, ehi_sub, fsky)

        cinv = _sim_cov_cinv(sim_cov_file, ibin, elo_sub, ehi_sub)
        A, sA, chi2, ndof, pval = _fit_null(cl_sub, cov, cl_gT_b, cinv=cinv)
        test_name = f"ell_range_{tag}"
        LOGGER.info("  ell_range %s %s z%d: A=%.3f±%.3f  chi2/dof=%.2f/%d  p=%.3f",
                    tag, tracer, ibin, A, sA, chi2, ndof, pval)
        rows.append({"test": test_name, "tracer": tracer, "ibin": ibin,
                     "A": A, "sigmaA": sA, "chi2": chi2, "ndof": ndof, "pvalue": pval})
    return rows


# ---------- LSS null tests ----------

def run_ngc_sgc(cfg, tracer, ibin, nside, ell_min, ell_max, delta_ell,
                apod_deg, apotype, apod_method, cmb_label, null_root,
                templates_dir, fsky_override, wsp_cache_root, overwrite):
    """
    Cross T with δ_NGC and δ_SGC separately. Tests sky-cap consistency.
    Requires per-cap delta maps (build with desi_build_maps.py --cap NGC/SGC).
    """
    results = []
    cmb_p = cmb_map_path(cfg, cmb_label, nside)
    mask_p = joint_mask_path(cfg, nside)

    for cap in ("NGC", "SGC"):
        delta_p = delta_map_path(cfg, tracer, ibin, nside, cap=cap)
        if not delta_p.exists():
            LOGGER.warning("[ngc_sgc] Per-cap delta map not found: %s", delta_p)
            LOGGER.warning("  Build with: python scripts/desi_build_maps.py --cap %s ...", cap)
            continue

        out = null_root / "ngc_sgc" / f"gT_{tracer}_cap{cap}_z{ibin}.npz"
        if out.exists() and not overwrite:
            r = _load_null_result(out, templates_dir, tracer, ibin, fsky_override)
            if r:
                r["test"] = f"ngc_sgc_{cap}"
                results.append(r)
            continue

        LOGGER.info("[ngc_sgc] %s %s z%d", cap, tracer, ibin)
        ell, elo, ehi, cl, fsky = _cross_spectrum(
            delta_p, cmb_p, mask_p, nside, ell_min, ell_max,
            delta_ell, apod_deg, apotype, apod_method, wsp_cache_root,
            wsp_tag_extra=f"_{cap.lower()}")

        _save_null_npz(out, ell, elo, ehi, cl, tracer, ibin, f"ngc_sgc_{cap}", nside, fsky,
                       cap=cap, cmb_label=cmb_label)

        fsky_use = fsky_override or fsky
        cl_gT_b, cov = _theory_for_bin(templates_dir, tracer, ibin, elo, ehi, fsky_use)
        A, sA, chi2, ndof, pval = _fit_null(cl, cov, cl_gT_b)
        LOGGER.info("  ngc_sgc %s %s z%d: A=%.3f±%.3f  chi2/dof=%.2f/%d",
                    cap, tracer, ibin, A, sA, chi2, ndof)
        results.append({"test": f"ngc_sgc_{cap}", "tracer": tracer, "ibin": ibin,
                         "A": A, "sigmaA": sA, "chi2": chi2, "ndof": ndof, "pvalue": pval})
    return results


def run_random_null(cfg, tracer, ibin, nside, ell_min, ell_max, delta_ell,
                    apod_deg, apotype, apod_method, cmb_label, null_root,
                    templates_dir, fsky_override, wsp_cache_root, overwrite):
    """
    Cross T with δ built from random catalog alone (should be zero signal).
    Requires a pre-built randoms-only overdensity map.
    Naming convention: delta_{tracer}_randoms_z{ibin}.fits.gz
    """
    delta_p = (_maps_root(cfg) / "desi" / tracer / f"nside{nside}" /
               f"delta_{tracer}_randoms_z{ibin}.fits.gz")
    if not delta_p.exists():
        LOGGER.warning("[random_null] Randoms-only delta map not found: %s", delta_p)
        LOGGER.warning("  Build by running desi_build_maps.py with randoms-only mode (TODO).")
        return None

    out = null_root / "random_null" / f"gT_{tracer}_z{ibin}.npz"
    if out.exists() and not overwrite:
        LOGGER.info("Skip existing %s", out)
        return _load_null_result(out, templates_dir, tracer, ibin, fsky_override)

    LOGGER.info("[random_null] %s z%d", tracer, ibin)
    cmb_p = cmb_map_path(cfg, cmb_label, nside)
    mask_p = lss_mask_path(cfg, tracer, nside)
    if not mask_p.exists():
        mask_p = joint_mask_path(cfg, nside)

    ell, elo, ehi, cl, fsky = _cross_spectrum(
        delta_p, cmb_p, mask_p, nside, ell_min, ell_max,
        delta_ell, apod_deg, apotype, apod_method, wsp_cache_root,
        wsp_tag_extra="_randoms")

    _save_null_npz(out, ell, elo, ehi, cl, tracer, ibin, "random_null", nside, fsky,
                   cmb_label=cmb_label)

    fsky_use = fsky_override or fsky
    cl_gT_b, cov = _theory_for_bin(templates_dir, tracer, ibin, elo, ehi, fsky_use)
    # Null expectation is zero; test chi² against zero
    chi2, ndof, pval = _chi2_null(cl, cov)
    LOGGER.info("  random_null %s z%d: chi2/dof=%.2f/%d  p=%.3f", tracer, ibin, chi2, ndof, pval)
    return {"test": "random_null", "tracer": tracer, "ibin": ibin, "A": float("nan"),
            "sigmaA": float("nan"), "chi2": chi2, "ndof": ndof, "pvalue": pval}


# ---------- helper to load a null result from an existing NPZ ----------

def _load_null_result(out_path, templates_dir, tracer, ibin, fsky_override, sim_cov_file=None):
    try:
        spec = np.load(out_path, allow_pickle=True)
        cl   = spec["cl"]
        elo  = spec["ell_edges_lo"]
        ehi  = spec["ell_edges_hi"]
        fsky = fsky_override or float(spec.get("fsky", 0.3))
        test = str(spec.get("null_test", "unknown"))
        cl_gT_b, cov = _theory_for_bin(templates_dir, tracer, ibin, elo, ehi, fsky)
        # sims cov only applies to full-CMB nulls (curl); not to comp_diff/ngc_sgc
        cinv = _sim_cov_cinv(sim_cov_file, ibin, elo, ehi) if test == "curl" else None
        A, sA, chi2, ndof, pval = _fit_null(cl, cov, cl_gT_b, cinv=cinv)
        return {"test": test, "tracer": tracer, "ibin": ibin,
                "A": A, "sigmaA": sA, "chi2": chi2, "ndof": ndof, "pvalue": pval}
    except Exception as e:
        LOGGER.warning("Could not load null result from %s: %s", out_path, e)
        return None


# ---------- main ----------

CMB_TESTS = ["curl", "comp_diff", "ell_range"]
LSS_TESTS  = ["ngc_sgc", "random_null"]


def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)

    spectra_root = _spectra_root(cfg)
    null_root = spectra_root / "nulls" / args.suite
    null_root.mkdir(parents=True, exist_ok=True)
    wsp_cache_root = spectra_root / "workspaces"

    tracers = args.tracers or list(cfg["desi"]["bins"].keys())
    bias_map = load_bias_priors(args.bias_priors)

    tests = args.tests or (CMB_TESTS if args.suite == "cmb" else LSS_TESTS)
    unknown = set(tests) - set(CMB_TESTS + LSS_TESTS)
    if unknown:
        LOGGER.error("Unknown test(s): %s", unknown)
        sys.exit(1)

    shared = dict(
        nside=args.nside, ell_min=args.ell_min, ell_max=args.ell_max,
        delta_ell=args.delta_ell, apod_deg=args.apod_deg, apotype=args.apotype,
        apod_method=args.apodizer, cmb_label=args.cmb_label,
        null_root=null_root, templates_dir=args.templates_dir,
        fsky_override=args.fsky, wsp_cache_root=wsp_cache_root,
        overwrite=args.overwrite,
    )

    all_rows = []

    for tracer in tracers:
        bins = cfg["desi"]["bins"].get(tracer, [])
        if not bins:
            LOGGER.warning("No bins configured for %s; skipping", tracer)
            continue

        # per-tracer sims covariance (supports a {TRACER} placeholder in the path)
        sim_cov_t = (args.sim_cov_file.format(TRACER=tracer)
                     if args.sim_cov_file else None)

        for ibin, (zmin, zmax) in enumerate(bins, start=1):
            LOGGER.info("── %s z%d [%.2f, %.2f] ──", tracer, ibin, zmin, zmax)

            if args.suite == "cmb":
                if "curl" in tests:
                    try:
                        r = run_curl(cfg, tracer, ibin, sim_cov_file=sim_cov_t, **shared)
                        if r: all_rows.append(r)
                    except (FileNotFoundError, Exception) as e:
                        LOGGER.error("[curl] %s z%d failed: %s", tracer, ibin, e)

                if "comp_diff" in tests:
                    try:
                        r = run_comp_diff(cfg, tracer, ibin, **{k: v for k, v in shared.items()
                                          if k not in ("cmb_label",)},
                                          label_a="SMICA", label_b="Commander")
                        if r: all_rows.append(r)
                    except (FileNotFoundError, Exception) as e:
                        LOGGER.error("[comp_diff] %s z%d failed: %s", tracer, ibin, e)

                if "ell_range" in tests:
                    try:
                        rows = run_ell_range(
                            cfg, tracer, ibin, args.ell_max, null_root,
                            args.templates_dir, args.fsky, bias_map, args.overwrite,
                            sim_cov_file=sim_cov_t)
                        all_rows.extend(rows)
                    except Exception as e:
                        LOGGER.error("[ell_range] %s z%d failed: %s", tracer, ibin, e)

            elif args.suite == "lss":
                if "ngc_sgc" in tests:
                    try:
                        rows = run_ngc_sgc(cfg, tracer, ibin, **shared)
                        all_rows.extend(rows)
                    except (FileNotFoundError, Exception) as e:
                        LOGGER.error("[ngc_sgc] %s z%d failed: %s", tracer, ibin, e)

                if "random_null" in tests:
                    try:
                        r = run_random_null(cfg, tracer, ibin, **shared)
                        if r: all_rows.append(r)
                    except (FileNotFoundError, Exception) as e:
                        LOGGER.error("[random_null] %s z%d failed: %s", tracer, ibin, e)

    # Write summary table
    out_csv = null_root / "null_table.csv"
    fieldnames = ["test", "tracer", "ibin", "A", "sigmaA", "chi2", "ndof", "pvalue"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: round(row[k], 6) if isinstance(row[k], float) else row[k]
                             for k in fieldnames})

    if all_rows:
        LOGGER.info("Wrote %d null results → %s", len(all_rows), out_csv)
        # Print a quick summary to stdout
        print(f"\n{'Test':<20} {'Tracer':>6} {'bin':>4}  {'A':>8}  {'σA':>8}  {'chi²':>7}  {'dof':>4}  {'p':>6}")
        print("-" * 75)
        for r in all_rows:
            A_str  = f"{r['A']:8.3f}"  if np.isfinite(r['A'])  else f"{'—':>8}"
            sA_str = f"{r['sigmaA']:8.3f}" if np.isfinite(r['sigmaA']) else f"{'—':>8}"
            print(f"{r['test']:<20} {r['tracer']:>6} {r['ibin']:>4}  {A_str}  {sA_str}"
                  f"  {r['chi2']:7.2f}  {r['ndof']:>4}  {r['pvalue']:6.3f}")
    else:
        LOGGER.warning("No null results — check inputs and logs above.")


if __name__ == "__main__":
    main()
