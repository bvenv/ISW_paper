#!/usr/bin/env python3
"""
Combine per-tracer/bin A_ISW measurements into the tomographic A_ISW(z) curve.

Reads the per-tracer tables written by fit_isw_amplitudes.py (via run_phase1_desi.py),
merges them, and produces:
  - results/desi_Aisw_combined.csv : all bins sorted by z + an inverse-variance combined row
  - results/desi_Aisw_combined.npz : arrays for downstream plotting
  - results/plots/Aisw_vs_z.png    : A_ISW(z) with error bars vs ΛCDM (A=1)

Combined amplitude is the inverse-variance weighted mean of the per-bin GLS estimates.
NOTE: this treats bins as independent. Tomographic bins share the same Planck T map, so
Cov(gT_i, gT_j) is non-zero; the combined error here is therefore a lower bound. The full
bin-bin covariance should come from the Gaussian sims (make_cmb_sims.py, step W5/W6).
"""
import argparse
import csv
import glob
import logging
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("combine_aisw")

# Consistent per-tracer colours for the curve
TRACER_COLORS = {"BGS": "#d62728", "LRG": "#1f77b4", "ELG": "#2ca02c", "QSO": "#9467bd"}


def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tables", default="results/aisw/*.csv",
                   help="Glob for per-tracer A_ISW CSVs (or a single merged CSV)")
    p.add_argument("--out-csv", default="results/desi_Aisw_combined.csv")
    p.add_argument("--out-npz", default="results/desi_Aisw_combined.npz")
    p.add_argument("--out-plot", default="results/plots/Aisw_vs_z.png")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def load_rows(tables_glob, bins_cfg):
    """Read all A_ISW rows; fill z from config bin edges if the table lacks them."""
    files = sorted(glob.glob(tables_glob))
    if not files:
        raise SystemExit(f"No A_ISW tables match {tables_glob}")
    rows = []
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    ibin = int(r["ibin"])
                except (ValueError, KeyError):
                    continue  # skip summary rows (e.g. "LRG_combined") from simcov tables
                tracer = r["tracer"]
                try:
                    zmin, zmax = float(r["zmin"]), float(r["zmax"])
                except (ValueError, KeyError):
                    zmin, zmax = bins_cfg[tracer][ibin - 1]
                rows.append({
                    "tracer": tracer, "ibin": ibin,
                    "zmin": zmin, "zmax": zmax, "zmid": 0.5 * (zmin + zmax),
                    "A": float(r["A"]), "sigmaA": float(r["sigmaA"]),
                })
    LOGGER.info("Loaded %d bins from %d table(s)", len(rows), len(files))
    return sorted(rows, key=lambda d: d["zmid"])


def inv_var_combine(A, sig):
    """Inverse-variance weighted mean and its error (bins assumed independent)."""
    A, sig = np.asarray(A, float), np.asarray(sig, float)
    good = np.isfinite(A) & np.isfinite(sig) & (sig > 0)
    if not good.any():
        return np.nan, np.nan
    w = 1.0 / sig[good] ** 2
    A_c = float(np.sum(w * A[good]) / np.sum(w))
    sig_c = float(1.0 / np.sqrt(np.sum(w)))
    return A_c, sig_c


def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = yaml.safe_load(open(args.config))
    rows = load_rows(args.tables, cfg["desi"]["bins"])

    A = np.array([r["A"] for r in rows])
    sig = np.array([r["sigmaA"] for r in rows])
    A_all, sig_all = inv_var_combine(A, sig)
    nsig_from_lcdm = (A_all - 1.0) / sig_all if sig_all > 0 else np.nan
    LOGGER.info("Combined A_ISW = %.3f ± %.3f  (%.1fσ from ΛCDM A=1, %.1fσ from 0)",
                A_all, sig_all, nsig_from_lcdm, A_all / sig_all if sig_all else np.nan)

    # per-tracer combined values
    tracers = sorted({r["tracer"] for r in rows})
    per_tracer = {}
    for t in tracers:
        sub = [r for r in rows if r["tracer"] == t]
        per_tracer[t] = inv_var_combine([r["A"] for r in sub], [r["sigmaA"] for r in sub])
        LOGGER.info("  %s: A_ISW = %.3f ± %.3f  (%d bins)", t, *per_tracer[t], len(sub))

    # ---- write merged CSV (+ combined rows) ----
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracer", "ibin", "zmin", "zmax", "zmid", "A", "sigmaA"])
        for r in rows:
            w.writerow([r["tracer"], r["ibin"], r["zmin"], r["zmax"],
                        round(r["zmid"], 4), round(r["A"], 6), round(r["sigmaA"], 6)])
        for t in tracers:
            w.writerow([f"{t}_combined", "", "", "", "",
                        round(per_tracer[t][0], 6), round(per_tracer[t][1], 6)])
        w.writerow(["ALL_combined", "", "", "", "", round(A_all, 6), round(sig_all, 6)])
    LOGGER.info("Wrote %s", out_csv)

    np.savez(args.out_npz,
             tracer=np.array([r["tracer"] for r in rows]),
             zmid=np.array([r["zmid"] for r in rows]),
             zmin=np.array([r["zmin"] for r in rows]),
             zmax=np.array([r["zmax"] for r in rows]),
             A=A, sigmaA=sig, A_combined=A_all, sigmaA_combined=sig_all)

    # ---- plot A_ISW(z) ----
    out_plot = Path(args.out_plot); out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(1.0, ls="--", lw=1.2, color="k", alpha=0.7, label=r"$\Lambda$CDM ($A=1$)")
    ax.axhline(0.0, ls=":", lw=1.0, color="gray", alpha=0.7, label="no ISW ($A=0$)")
    for t in tracers:
        sub = [r for r in rows if r["tracer"] == t]
        zc = [r["zmid"] for r in sub]
        xerr = [[r["zmid"] - r["zmin"] for r in sub], [r["zmax"] - r["zmid"] for r in sub]]
        ax.errorbar(zc, [r["A"] for r in sub], yerr=[r["sigmaA"] for r in sub], xerr=xerr,
                    fmt="o", capsize=3, color=TRACER_COLORS.get(t, "k"), label=t)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"$A_{\rm ISW}$")
    ax.set_title(rf"Tomographic ISW amplitude  (combined $A={A_all:.2f}\pm{sig_all:.2f}$)")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(out_plot, dpi=160); plt.close(fig)
    LOGGER.info("Wrote %s", out_plot)


if __name__ == "__main__":
    main()
