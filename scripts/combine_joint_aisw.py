#!/usr/bin/env python3
"""
Multi-tracer A_ISW combination, two ways:

  "naive A"  — inverse-variance weighted mean of the per-bin A_ISW estimates (bins treated
               independent). Quick; the headline A_ISW(z) points come from here.
  "joint A"  — a SINGLE global amplitude fit across ALL tracer×bin bandpowers on the shared
               theory curve, via GLS using each tracer's full sims bandpower covariance
               (block-diagonal across tracers — cross-tracer terms neglected, a lower bound).
               This is the optimal single-amplitude estimator (theory-shape weighting).

Bins can be dropped with --drop TRACER:ibin (e.g. the near-empty ELG z1).

Outputs results/desi_Aisw_joint.csv and results/plots/Aisw_vs_z_joint.png.
"""
import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from fit_isw_amplitudes import bandaverage

TRACER_COLORS = {"BGS": "#d62728", "LRG": "#1f77b4", "ELG": "#2ca02c", "QSO": "#9467bd"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--templates-dir", default="templates/isw")
    p.add_argument("--simcov-glob", default="results/sim_cov_*.npz")
    p.add_argument("--aisw-glob", default="results/desi_Aisw_*_simcov.csv")
    p.add_argument("--drop", nargs="*", default=[], help="Drop bins, e.g. ELG:1 QSO:3")
    p.add_argument("--out-csv", default="results/desi_Aisw_joint.csv")
    p.add_argument("--out-plot", default="results/plots/Aisw_vs_z_joint.png")
    return p.parse_args()


def inv_var(A, sig):
    A, sig = np.asarray(A, float), np.asarray(sig, float)
    w = 1.0 / sig ** 2
    return float(np.sum(w * A) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    bins_cfg = cfg["desi"]["bins"]
    spectra = Path(cfg["paths"]["spectra"]) / "desi"
    drop = {(d.split(":")[0], int(d.split(":")[1])) for d in args.drop}
    L = args.ell_max

    # ---- per-bin points (for the naive combine + the plot) ----
    pts = []  # (tracer, ibin, zmid, A, sigmaA)
    for f in sorted(glob.glob(args.aisw_glob)):
        for r in csv.DictReader(open(f)):
            try:
                ib = int(r["ibin"])
            except (ValueError, KeyError):
                continue
            T = r["tracer"]
            if (T, ib) in drop:
                continue
            zmin, zmax = bins_cfg[T][ib - 1]
            pts.append((T, ib, 0.5 * (zmin + zmax), float(r["A"]), float(r["sigmaA"])))

    A_naive, sig_naive = inv_var([p[3] for p in pts], [p[4] for p in pts])

    # ---- joint A: GLS combine of the per-bin amplitudes with the nbin×nbin amplitude
    # covariance (cross-tracer included), from sim_covariance_joint.py. Each per-bin A
    # estimates the same global amplitude (=1 in ΛCDM), so A_joint = (1ᵀΣ⁻¹A)/(1ᵀΣ⁻¹1). ----
    sc = np.load("results/sim_cov_all.npz", allow_pickle=True)
    Sigma, A_data, nsims = sc["Sigma"], sc["A_data"], int(sc["nsims"])
    tr_lay, ib_lay = [str(x) for x in sc["tracers"]], [int(x) for x in sc["ibins"]]
    keep = [j for j, (T, ib) in enumerate(zip(tr_lay, ib_lay)) if (T, ib) not in drop]

    def gls(idx):
        S = Sigma[np.ix_(idx, idx)]
        Sinv = (nsims - len(idx) - 2) / (nsims - 1) * np.linalg.inv(S)
        one = np.ones(len(idx)); den = float(one @ Sinv @ one)
        return float((one @ Sinv @ A_data[idx]) / den), float(1 / np.sqrt(den))

    A_joint, sig_joint = gls(keep)
    per_tracer_joint = {}
    for T in sorted(set(tr_lay)):
        ti = [j for j in keep if tr_lay[j] == T]
        if ti:
            per_tracer_joint[T] = gls(ti)

    # ---- bonus: smooth evolution fit  A_ISW(z) = A0 + A1*(z - zpiv)  (GLS, same Σ) ----
    z_keep = np.array([0.5 * sum(bins_cfg[tr_lay[j]][ib_lay[j] - 1]) for j in keep])
    Sinv_k = (nsims - len(keep) - 2) / (nsims - 1) * np.linalg.inv(Sigma[np.ix_(keep, keep)])
    one = np.ones(len(keep))
    zpiv = float((one @ Sinv_k @ z_keep) / (one @ Sinv_k @ one))  # pivot decorrelates A0, A1
    M = np.column_stack([one, z_keep - zpiv])
    covp = np.linalg.inv(M.T @ Sinv_k @ M)
    p = covp @ M.T @ Sinv_k @ A_data[keep]                        # [A0=A(zpiv), A1=dA/dz]
    A0, A1, sA0, sA1 = float(p[0]), float(p[1]), float(np.sqrt(covp[0, 0])), float(np.sqrt(covp[1, 1]))

    # ---- report ----
    print(f"\nMulti-tracer A_ISW  (dropped: {sorted(drop) or 'none'})")
    print(f"  naive A (inv-var, {len(pts)} bins): {A_naive:.3f} ± {sig_naive:.3f}  "
          f"({(A_naive-1)/sig_naive:+.1f}σ from ΛCDM, {A_naive/sig_naive:.1f}σ from 0)")
    print(f"  joint A (shared curve, cross-tracer amplitude cov): {A_joint:.3f} ± {sig_joint:.3f}  "
          f"({(A_joint-1)/sig_joint:+.1f}σ from ΛCDM, {A_joint/sig_joint:.1f}σ from 0)")
    for T, (a, s) in sorted(per_tracer_joint.items()):
        print(f"    {T} joint: {a:.2f} ± {s:.2f}")
    print(f"  evolution fit: A({zpiv:.2f}) = {A0:.2f} ± {sA0:.2f},  dA/dz = {A1:+.2f} ± {sA1:.2f}  "
          f"({A1/sA1:+.1f}σ slope — {'no evolution' if abs(A1/sA1) < 2 else 'EVOLUTION'} )")

    with open(args.out_csv, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["tracer", "ibin", "zmid", "A", "sigmaA"])
        for T, ib, z, A, s in pts:
            w.writerow([T, ib, round(z, 3), round(A, 5), round(s, 5)])
        w.writerow(["NAIVE_combined", "", "", round(A_naive, 5), round(sig_naive, 5)])
        w.writerow(["JOINT_combined", "", "", round(A_joint, 5), round(sig_joint, 5)])
        w.writerow(["EVOL_A0(zpiv)", round(zpiv, 3), "", round(A0, 5), round(sA0, 5)])
        w.writerow(["EVOL_dAdz", "", "", round(A1, 5), round(sA1, 5)])

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(1, ls="--", lw=1.2, c="k", alpha=.7, label=r"$\Lambda$CDM ($A=1$)")
    ax.axhline(0, ls=":", lw=1, c="gray", alpha=.7)
    for T in sorted({p[0] for p in pts}):
        sub = [p for p in pts if p[0] == T]
        ax.errorbar([p[2] for p in sub], [p[3] for p in sub], yerr=[p[4] for p in sub],
                    fmt="o", capsize=3, color=TRACER_COLORS.get(T, "k"), label=T)
    ax.axhspan(A_joint - sig_joint, A_joint + sig_joint, color="orange", alpha=.18,
               label=f"joint $A={A_joint:.2f}\\pm{sig_joint:.2f}$")
    ax.axhline(A_joint, color="orange", lw=1.5)
    zg = np.linspace(min(p[2] for p in pts) - 0.05, max(p[2] for p in pts) + 0.05, 50)
    ax.plot(zg, A0 + A1 * (zg - zpiv), "--", color="purple", lw=1.5,
            label=f"evolution: $dA/dz={A1:+.2f}\\pm{sA1:.2f}$")
    ax.set_xlabel("redshift $z$"); ax.set_ylabel(r"$A_{\rm ISW}$")
    ax.set_title(f"DESI×Planck tomographic ISW (4 tracers)   joint $A={A_joint:.2f}\\pm{sig_joint:.2f}$"
                 f"  ({A_joint/sig_joint:.1f}$\\sigma$)")
    ax.grid(ls=":", alpha=.4); ax.legend(ncol=3, fontsize=8, frameon=False)
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out_plot, dpi=150); plt.close(fig)
    print(f"  wrote {args.out_csv} + {args.out_plot}\n")


if __name__ == "__main__":
    main()
