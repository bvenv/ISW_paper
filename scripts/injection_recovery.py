#!/usr/bin/env python3
"""
Sensitivity / validation test: can the pipeline recover an EVOLVING A_ISW(z) (e.g. a
modified-gravity departure), or is the flat ΛCDM-consistent result just insensitivity?

Using the measured cross-tracer amplitude covariance Σ (results/sim_cov_all.npz), we inject a
known A_ISW(z) = A0 + s·(z − z_piv), add a noise draw ~ N(0, Σ), and refit (A0, s) with the same
GLS used on the data. We show (a) recovery is unbiased over a grid of injected slopes, and
(b) one representative "evolving ISW" injection vs its recovered band — i.e. what the method
WOULD see. Reports the DR1 sensitivity (σ on the slope/amplitude → the deviation detectable at 3σ).

This validates the null result; it is NOT a modified-gravity constraint (DR1 S/N only bounds
large deviations — the full survey is where MG becomes competitive).
"""
import argparse
from pathlib import Path
import numpy as np
import yaml
import matplotlib; matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--simcov", default="results/sim_cov_all.npz")
    p.add_argument("--nmock", type=int, default=2000)
    p.add_argument("--demo-slope", type=float, default=2.0, help="Injected dA/dz for the demo panel")
    p.add_argument("--out-plot", default="results/plots/injection_recovery.png")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--drop", nargs="*", default=[], help="Drop bins, e.g. ELG:1 ELG:3")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    bins = cfg["desi"]["bins"]
    sc = np.load(a.simcov, allow_pickle=True)
    Sigma = sc["Sigma"]; nsims = int(sc["nsims"])
    tr = [str(x) for x in sc["tracers"]]; ib = [int(x) for x in sc["ibins"]]
    drop = {(d.split(":")[0], int(d.split(":")[1])) for d in a.drop}
    keep = [j for j, (t, i) in enumerate(zip(tr, ib)) if (t, i) not in drop]
    Sigma = Sigma[np.ix_(keep, keep)]
    tr = [tr[j] for j in keep]; ib = [ib[j] for j in keep]
    z = np.array([0.5 * sum(bins[tr[j]][ib[j] - 1]) for j in range(len(tr))])
    n = len(z)

    Sinv = (nsims - n - 2) / (nsims - 1) * np.linalg.inv(Sigma)   # Hartlap
    one = np.ones(n)
    zpiv = float((one @ Sinv @ z) / (one @ Sinv @ one))
    M = np.column_stack([one, z - zpiv])
    Fisher = M.T @ Sinv @ M
    covp = np.linalg.inv(Fisher)                                  # [A0, slope] covariance
    sig_A0, sig_s = float(np.sqrt(covp[0, 0])), float(np.sqrt(covp[1, 1]))

    def fit(A_vec):
        return covp @ M.T @ Sinv @ A_vec                         # [A0, slope]

    rng = np.random.default_rng(a.seed)
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(n))

    # (a) unbiased recovery over a grid of injected slopes
    slopes = np.linspace(-3, 3, 13)
    rec_mean, rec_std = [], []
    for s in slopes:
        truth = 1.0 + s * (z - zpiv)
        recs = np.array([fit(truth + L @ rng.standard_normal(n))[1] for _ in range(a.nmock)])
        rec_mean.append(recs.mean()); rec_std.append(recs.std())
    rec_mean, rec_std = np.array(rec_mean), np.array(rec_std)

    print(f"\nInjection-recovery (n={n} bins, {a.nmock} mocks)")
    print(f"  Fisher sensitivity:  sigma(A0) = {sig_A0:.2f},  sigma(dA/dz) = {sig_s:.2f}")
    print(f"  3σ-detectable: amplitude deviation |A-1| > {3*sig_A0:.1f};  slope |dA/dz| > {3*sig_s:.1f}")
    print(f"  recovery unbiased: <rec slope - injected> = {np.mean(rec_mean - slopes):+.3f} "
          f"(<< sigma_s={sig_s:.2f})")

    # (b) one representative evolving injection + its recovered band
    truth_demo = 1.0 + a.demo_slope * (z - zpiv)
    A_obs = truth_demo + L @ rng.standard_normal(n)
    p_rec = fit(A_obs); A0r, sr = float(p_rec[0]), float(p_rec[1])
    zg = np.linspace(z.min() - 0.05, z.max() + 0.05, 50)
    band = np.sqrt(covp[0, 0] + (zg - zpiv) ** 2 * covp[1, 1] + 2 * (zg - zpiv) * covp[0, 1])
    print(f"  demo: injected dA/dz={a.demo_slope:+.1f} -> recovered {sr:+.2f} ± {sig_s:.2f} "
          f"({sr/sig_s:.1f}σ from flat)")

    # ---- forward-looking sensitivity: errors scale ~1/sqrt(area) until the cosmic-variance
    # ceiling (total galaxy-ISW S/N ~7-8σ -> sigma(A) floor ~0.13). MG models typically alter
    # the ISW amplitude at the O(10-30%) level, so the floor is where galaxy ISW becomes MG-relevant.
    A0_FLOOR = 0.13                                   # full-sky/all-tracer cosmic-variance limit on A
    releases = [("DESI DR1 (this work)", 1.0), ("DESI DR2 (~2x area)", 2.0),
                ("DESI Y5 (~3x area)", 3.0), ("cosmic-variance limit", None)]
    print("  projected sensitivity (errors ~ 1/sqrt(area), floored at the CV limit):")
    for name, mult in releases:
        if mult is None:
            sa, ss = A0_FLOOR, sig_s * A0_FLOOR / sig_A0
        else:
            sa, ss = max(sig_A0 / np.sqrt(mult), A0_FLOOR), sig_s / np.sqrt(mult)
        print(f"    {name:24s} sigma(A)~{sa:.2f}  sigma(dA/dz)~{ss:.2f}  "
              f"(detects |1-A|>{3*sa:.0%} at 3σ)")
    print("  -> MG ISW deviations are O(10-30%); galaxy ISW reaches that only near the CV limit "
          "(full DESI + full-sky), so the bispectrum/other probes carry the MG power meanwhile.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 8.6))
    ax1.errorbar(slopes, rec_mean, yerr=rec_std, fmt="o", capsize=3, color="C0", label="recovered (mock mean ± scatter)")
    ax1.plot(slopes, slopes, "k--", lw=1, label="unbiased (1:1)")
    ax1.axhspan(-3 * sig_s, 3 * sig_s, color="gray", alpha=.12, label=r"$|dA/dz|<3\sigma$ (undetectable)")
    ax1.set_xlabel("injected $dA/dz$"); ax1.set_ylabel("recovered $dA/dz$")
    ax1.set_title(f"Unbiased recovery  ($\\sigma_{{dA/dz}}={sig_s:.1f}$)")
    ax1.grid(ls=":", alpha=.4); ax1.legend(fontsize=8, frameon=False)

    ax2.axhline(1, ls="--", c="k", alpha=.6, label=r"$\Lambda$CDM (flat)")
    ax2.plot(zg, 1 + a.demo_slope * (zg - zpiv), "-", color="purple", lw=2, label=f"injected ($dA/dz={a.demo_slope:+.0f}$)")
    ax2.errorbar(z, A_obs, yerr=np.sqrt(np.diag(Sigma)), fmt="o", ms=4, color="C1", alpha=.6, capsize=2, label="mock data")
    ax2.plot(zg, A0r + sr * (zg - zpiv), "-", color="C0", lw=1.8, label=f"recovered ($dA/dz={sr:+.1f}\\pm{sig_s:.1f}$)")
    ax2.fill_between(zg, A0r + sr * (zg - zpiv) - band, A0r + sr * (zg - zpiv) + band, color="C0", alpha=.18)
    ax2.set_xlabel("redshift $z$"); ax2.set_ylabel(r"$A_{\rm ISW}$")
    ax2.set_title("Representative evolving / MG-like injection")
    ax2.grid(ls=":", alpha=.4); ax2.legend(fontsize=8, frameon=False)

    Path(a.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out_plot, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {a.out_plot}\n")


if __name__ == "__main__":
    main()
