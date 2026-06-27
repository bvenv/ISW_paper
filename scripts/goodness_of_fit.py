#!/usr/bin/env python3
"""
Global goodness-of-fit of the tomographic ISW amplitude vector A_ISW(z) against (a) ΛCDM (all
A=1) and (b) the no-signal hypothesis (all A=0), using the cross-tracer amplitude covariance Σ
(results/sim_cov_all.npz). Gives the quantitative backing for "consistent with ΛCDM":

    chi2_LCDM = (A-1)^T Σ^-1 (A-1)          -> PTE for consistency with ΛCDM
    chi2_null = A^T Σ^-1 A                   -> consistency with zero
    Δchi2 = chi2_null - chi2_LCDM            -> global detection significance sqrt(Δchi2)

Reported for all bins and with the null-test-flagged ELG z3 dropped.
"""
import argparse
import numpy as np
import yaml
from scipy.stats import chi2 as chi2_dist


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--simcov", default="results/sim_cov_all.npz")
    p.add_argument("--drop", nargs="*", default=["ELG:3"], help="bins to drop, e.g. ELG:3")
    p.add_argument("--out-csv", default="results/desi_isw_gof.csv")
    return p.parse_args()


def gof(A, Sigma, nsims, label):
    n = len(A)
    Sinv = (nsims - n - 2) / (nsims - 1) * np.linalg.inv(Sigma)      # Hartlap
    one = np.ones(n)
    chi2_l = float((A - one) @ Sinv @ (A - one))
    chi2_0 = float(A @ Sinv @ A)
    dchi2 = chi2_0 - chi2_l
    pte_l = float(chi2_dist.sf(chi2_l, n))
    print(f"  [{label}]  n={n}")
    print(f"    vs ΛCDM (A=1):  chi2/dof = {chi2_l:.1f}/{n} = {chi2_l/n:.2f}   PTE = {pte_l:.3f}"
          f"   ({'consistent' if pte_l > 0.05 else 'TENSION'})")
    print(f"    vs zero (A=0):  chi2     = {chi2_0:.1f}   -> global detection "
          f"sqrt(Δchi2) = {np.sqrt(max(dchi2,0)):.1f}σ")
    return dict(label=label, n=n, chi2_lcdm=chi2_l, dof=n, pte_lcdm=pte_l,
                chi2_null=chi2_0, detect_sigma=float(np.sqrt(max(dchi2, 0))))


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    sc = np.load(a.simcov, allow_pickle=True)
    Sigma, A_data, nsims = sc["Sigma"], sc["A_data"], int(sc["nsims"])
    tr = [str(x) for x in sc["tracers"]]; ib = [int(x) for x in sc["ibins"]]
    drop = {(d.split(":")[0], int(d.split(":")[1])) for d in a.drop}

    print("\nGlobal ISW goodness-of-fit (amplitude vector vs hypotheses)")
    rows = [gof(A_data, Sigma, nsims, "all bins")]
    keep = [j for j, (T, i) in enumerate(zip(tr, ib)) if (T, i) not in drop]
    if len(keep) < len(tr):
        rows.append(gof(A_data[keep], Sigma[np.ix_(keep, keep)], nsims,
                        f"drop {sorted(drop)}"))
    print()

    import csv
    with open(a.out_csv, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {a.out_csv}\n")


if __name__ == "__main__":
    main()
