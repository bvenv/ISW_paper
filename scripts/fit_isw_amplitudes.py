#!/usr/bin/env python3
"""
Fit A_ISW per redshift bin using precomputed g×T spectra and a theory template.
Implements fast GLS estimator and (optional) emcee MCMC.

Inputs
------
- spectra/desi/gT_{TRACER}_z{ibin}_lmax{L}.npz   (from compute_crosscls.py)
- templates/isw/gT_{TRACER}.npz                   (from build_isw_templates_clank.py)
- templates/isw/gg_{TRACER}.npz
- templates/isw/cltt_camb_uK2.txt
- results/desi_bias_priors.json                    (from calibrate_bias.py)

Outputs
-------
- results/desi_Aisw_table.csv
- results/posterior_{TRACER}_z{ibin}.npy           (if --do-mcmc)
"""
import argparse
import csv
import glob
import json
import logging
from pathlib import Path

import numpy as np
import yaml

LOGGER = logging.getLogger("fit_isw_amplitudes")


def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--spectra-glob", default="spectra/desi/gT_*_lmax*.npz",
                   help="Glob for input gxT bandpower NPZs")
    p.add_argument("--templates-dir", default="templates/isw",
                   help="Directory containing theory template NPZs")
    p.add_argument("--bias-priors", default="results/desi_bias_priors.json")
    p.add_argument("--fsky", type=float, default=None,
                   help="Override sky fraction for Gaussian covariance "
                        "(default: compute from mask stored in spectra NPZ, then config fsky)")
    p.add_argument("--outfile", default="results/desi_Aisw_table.csv")
    p.add_argument("--do-mcmc", action="store_true",
                   help="Run emcee MCMC per bin after GLS; saves posterior_{TRACER}_z{ibin}.npy")
    p.add_argument("--mcmc-nsteps", type=int, default=2000)
    p.add_argument("--mcmc-nwalkers", type=int, default=32)
    p.add_argument("--mcmc-burnin", type=int, default=500)
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------- theory templates ----------

def load_theory_templates(templates_dir: str, tracer: str, ibin: int):
    """
    Return (ell, cl_gT, cl_gg, cl_TT) on the template ell grid (0..lmax_template).

    Tries per-bin files first (gT_{TRACER}_z{ibin}.npz), falls back to
    tracer-level (gT_{TRACER}.npz). cl_gT and cl_TT are in µK²; cl_gg is
    dimensionless (includes fiducial b²).
    """
    tdir = Path(templates_dir)

    def _npz(name):
        path = tdir / name
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path)

    # gT
    try:
        d_gT = _npz(f"gT_{tracer}_z{ibin}.npz")
    except FileNotFoundError:
        d_gT = _npz(f"gT_{tracer}.npz")
        LOGGER.debug("No per-bin gT template for %s z%d; using tracer-level", tracer, ibin)

    # gg
    try:
        d_gg = _npz(f"gg_{tracer}_z{ibin}.npz")
    except FileNotFoundError:
        d_gg = _npz(f"gg_{tracer}.npz")
        LOGGER.debug("No per-bin gg template for %s z%d; using tracer-level", tracer, ibin)

    ell = d_gT["ell"].astype(int)
    cl_gT = d_gT["cl"].copy()
    cl_gg = d_gg["cl"].copy()

    # TT: two-column text (ell, C_ell µK²)
    cltt_arr = np.loadtxt(tdir / "cltt_camb_uK2.txt")
    cl_TT = np.interp(ell, cltt_arr[:, 0], cltt_arr[:, 1], left=0.0, right=0.0)

    return ell, cl_gT, cl_gg, cl_TT


def bandaverage(ell_th, cl_th, ell_lo, ell_hi):
    """(2ℓ+1)-weighted average of cl_th into each NaMaster bandpower bin."""
    cl_band = np.zeros(len(ell_lo))
    for b in range(len(ell_lo)):
        lo, hi = int(ell_lo[b]), int(ell_hi[b])
        sel = (ell_th >= lo) & (ell_th <= hi)
        if sel.sum() == 0:
            cl_band[b] = np.interp(0.5 * (lo + hi), ell_th, cl_th)
        else:
            w = 2 * ell_th[sel] + 1
            cl_band[b] = np.dot(w, cl_th[sel]) / w.sum()
    return cl_band


# ---------- covariance ----------

def gaussian_cov(cl_gT_th, cl_gg_th, cl_TT_th, ell_lo, ell_hi, fsky):
    """
    Diagonal Gaussian bandpower covariance for C_ell^{gT}.

    Var(Ĉ_b^{gT}) = [C_b^{gT,th}² + C_b^{gg,th} * C_b^{TT,th}]
                    / (f_sky * Σ_{ℓ∈b} (2ℓ+1))
    """
    var = np.zeros(len(cl_gT_th))
    for b in range(len(var)):
        ells = np.arange(int(ell_lo[b]), int(ell_hi[b]) + 1)
        n_modes = fsky * np.sum(2 * ells + 1)
        var[b] = (cl_gT_th[b] ** 2 + cl_gg_th[b] * cl_TT_th[b]) / n_modes
    return np.diag(var)


def fsky_from_mask(mask_path: str) -> float | None:
    try:
        import healpy as hp
        mask = hp.read_map(mask_path, verbose=False)
        return float(np.mean(mask > 0))
    except Exception as e:
        LOGGER.warning("Could not load mask %s for fsky: %s", mask_path, e)
        return None


# ---------- estimators ----------

def fit_gls(cl_data, cov, cl_theory):
    """
    GLS amplitude and 1-sigma uncertainty.
    A = (t C⁻¹ d) / (t C⁻¹ t),  σ_A = 1 / sqrt(t C⁻¹ t)
    """
    C_inv = np.linalg.inv(cov)
    tCt = cl_theory @ C_inv @ cl_theory
    tCd = cl_theory @ C_inv @ cl_data
    return float(tCd / tCt), float(1.0 / np.sqrt(tCt))


def run_mcmc(cl_data, cov, cl_theory, out_path: Path, nwalkers, nsteps, burnin):
    """Sample the 1-D A_ISW posterior with emcee. Returns (A_median, sigma)."""
    try:
        import emcee
    except ImportError:
        LOGGER.warning("emcee not installed; skipping MCMC (pip install emcee)")
        return None, None

    C_inv = np.linalg.inv(cov)
    A_gls, sigma_gls = fit_gls(cl_data, cov, cl_theory)

    def log_prob(theta):
        A = theta[0]
        if not (-20.0 < A < 20.0):
            return -np.inf
        r = cl_data - A * cl_theory
        return -0.5 * r @ C_inv @ r

    rng = np.random.default_rng()
    p0 = A_gls + sigma_gls * 0.1 * rng.standard_normal((nwalkers, 1))
    sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)

    chain = sampler.get_chain(discard=burnin, flat=True)[:, 0]
    np.save(out_path, chain)
    A_med = float(np.median(chain))
    sigma = float(np.std(chain))
    LOGGER.info("  MCMC: A = %.3f ± %.3f  → %s", A_med, sigma, out_path)
    return A_med, sigma


# ---------- helpers ----------

def load_bias_priors(path: str) -> dict:
    """Return dict keyed (tracer, ibin) → entry dict from JSON priors file."""
    if not Path(path).exists():
        LOGGER.warning("Bias priors not found: %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    return {(e["tracer"], e["ibin"]): e for e in data.get("priors", [])}


def parse_npz_name(path: Path):
    """
    Parse tracer and ibin from filenames like gT_LRG_z2_lmax150.npz.
    Returns (tracer, ibin) or (None, None) on failure.
    """
    parts = path.stem.split("_")  # ['gT', 'LRG', 'z2', 'lmax150']
    if len(parts) < 3:
        return None, None
    tracer = parts[1]
    for part in parts[2:]:
        if part.startswith("z") and part[1:].isdigit():
            return tracer, int(part[1:])
    return tracer, None


# ---------- main ----------

def main():
    args = parse_args()
    setup_logging(args.verbose)

    cfg = load_config(args.config)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    bias_map = load_bias_priors(args.bias_priors)

    spectra_files = sorted(glob.glob(args.spectra_glob))
    if not spectra_files:
        LOGGER.warning("No spectra files found matching: %s", args.spectra_glob)

    rows = []
    for fpath in spectra_files:
        p = Path(fpath)
        tracer, ibin = parse_npz_name(p)
        if tracer is None or ibin is None:
            LOGGER.warning("Cannot parse tracer/ibin from %s; skipping", p.name)
            continue

        LOGGER.info("Fitting %s bin %d  (%s)", tracer, ibin, p.name)
        spec = np.load(fpath, allow_pickle=True)
        cl_data = spec["cl"]
        ell_lo = spec["ell_edges_lo"]
        ell_hi = spec["ell_edges_hi"]

        # sky fraction: CLI → mask in NPZ → config → hardcoded fallback
        fsky = args.fsky
        if fsky is None and "mask_path" in spec:
            fsky = fsky_from_mask(str(spec["mask_path"]))
        if fsky is None:
            fsky = cfg.get("estimator", {}).get("fsky", 0.3)
            LOGGER.debug("fsky not computed from mask; using %.3f", fsky)

        # Load and band-average theory
        try:
            ell_th, cl_gT_th, cl_gg_th, cl_TT_th = load_theory_templates(
                args.templates_dir, tracer, ibin)
        except FileNotFoundError as e:
            LOGGER.error("Missing template: %s — skipping %s bin %d", e, tracer, ibin)
            continue

        cl_gT_band = bandaverage(ell_th, cl_gT_th, ell_lo, ell_hi)
        cl_gg_band = bandaverage(ell_th, cl_gg_th, ell_lo, ell_hi)
        cl_TT_band = bandaverage(ell_th, cl_TT_th, ell_lo, ell_hi)

        cov = gaussian_cov(cl_gT_band, cl_gg_band, cl_TT_band, ell_lo, ell_hi, fsky)

        A, sigma_A = fit_gls(cl_data, cov, cl_gT_band)
        LOGGER.info("  GLS: A_ISW = %.3f ± %.3f", A, sigma_A)

        if args.do_mcmc:
            mcmc_out = outfile.parent / f"posterior_{tracer}_z{ibin}.npy"
            A_mc, sigma_mc = run_mcmc(
                cl_data, cov, cl_gT_band, mcmc_out,
                nwalkers=args.mcmc_nwalkers,
                nsteps=args.mcmc_nsteps,
                burnin=args.mcmc_burnin,
            )
            if A_mc is not None:
                A, sigma_A = A_mc, sigma_mc

        prior = bias_map.get((tracer, ibin), {})
        rows.append({
            "tracer": tracer,
            "ibin": ibin,
            "zmin": prior.get("zmin", ""),
            "zmax": prior.get("zmax", ""),
            "A": round(A, 6),
            "sigmaA": round(sigma_A, 6),
            "lmin": int(ell_lo[0]),
            "lmax": int(ell_hi[-1]),
            "fsky": round(fsky, 4),
            "estimator": "mcmc" if args.do_mcmc else "gls",
        })

    fieldnames = ["tracer", "ibin", "zmin", "zmax", "A", "sigmaA", "lmin", "lmax", "fsky", "estimator"]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        LOGGER.info("Wrote %d rows → %s", len(rows), outfile)
    else:
        LOGGER.warning("No bins fitted — check spectra glob and templates.")


if __name__ == "__main__":
    main()
