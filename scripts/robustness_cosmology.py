#!/usr/bin/env python3
"""
Fiducial-cosmology robustness of A_ISW.

Rebuilds the gT ISW template at the fiducial cosmology and at Planck ±1σ in Ωm and σ₈ (holding
n(z) and the κ-pinned bias fixed — this isolates the template's cosmology dependence, the actual
referee question), refits A_ISW per bin against the SAME measured bandpowers and sims covariance,
and recombines the joint A via the cross-tracer amplitude covariance Σ. Reports the joint A at
each cosmology; the spread vs σ_A is the robustness.

A_ISW is an amplitude relative to the assumed template, so this answers: does the conclusion move
if we assumed a (Planck-)different fiducial cosmology? Templates: CAMB source windows (reused).
"""
import argparse, glob
from pathlib import Path
import numpy as np
import yaml

from build_isw_templates_clank import camb_window_cls, load_bias_priors, BIAS_FALLBACK
from fit_isw_amplitudes import bandaverage

# Planck 2018 TT,TE,EE+lowE+lensing ±1σ
OM, OM_SIG = 0.3153, 0.0073
S8, S8_SIG = 0.8111, 0.0060


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--bias-priors", default="results/desi_bias_kappa.json")
    p.add_argument("--spectra-root", default=None, help="default: paths.spectra/desi")
    p.add_argument("--lmax", type=int, default=150)
    p.add_argument("--out-csv", default="results/cosmology_robustness.csv")
    return p.parse_args()


def fid_cosmo(cfg):
    c = cfg["theory"]["cosmo"]
    return dict(H0=float(c["H0"]), ombh2=float(c["ombh2"]), omch2=float(c["omch2"]),
                ns=float(c["ns"]), As=float(c["As"]), tau=float(c["tau"]))


def variants(cp):
    """Map Ωm and σ₈ ±1σ onto CAMB params (hold H0, ombh2, ns): omch2 sets Ωm, As ∝ σ₈²."""
    h2 = (cp["H0"] / 100) ** 2
    out = {"fiducial": dict(cp)}
    for sign, tag in [(+1, "+"), (-1, "-")]:
        om = dict(cp); om["omch2"] = cp["omch2"] + sign * OM_SIG * h2   # δΩm·h² into CDM
        out[f"Om{tag}1sig"] = om
        s8 = dict(cp); s8["As"] = cp["As"] * (1 + sign * S8_SIG / S8) ** 2  # As ∝ σ8²
        out[f"s8{tag}1sig"] = s8
    return out


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    cp = fid_cosmo(cfg)
    bins_cfg = cfg["desi"]["bins"]
    spectra = Path(a.spectra_root or (Path(cfg["paths"]["spectra"]) / "desi"))
    biasmap = load_bias_priors(a.bias_priors)

    sc = np.load("results/sim_cov_all.npz", allow_pickle=True)
    Sigma = sc["Sigma"]; nsims = int(sc["nsims"])
    tr_lay = [str(x) for x in sc["tracers"]]; ib_lay = [int(x) for x in sc["ibins"]]

    # per-tracer bandpower covariances (block-diagonal per bin)
    covs = {}
    for T in set(tr_lay):
        f = f"results/sim_cov_{T}.npz"
        if Path(f).exists():
            d = np.load(f, allow_pickle=True)
            covs[T] = (d["cov"], int(d["nband"]))

    def bin_block(T, ib):
        cov, nb = covs[T]
        sl = slice((ib - 1) * nb, ib * nb)
        return cov[sl, sl]

    def A_perbin(cosmo):
        """Recompute A_ISW for every tracer×bin with the template at this cosmology."""
        A = np.full(len(tr_lay), np.nan)
        for j, (T, ib) in enumerate(zip(tr_lay, ib_lay)):
            sp = spectra / f"gT_{T}_z{ib}_lmax{a.lmax}.npz"
            if not sp.exists():
                continue
            spec = np.load(sp, allow_pickle=True)
            d = spec["cl"]; lo, hi = spec["ell_edges_lo"], spec["ell_edges_hi"]
            z, nz = np.loadtxt(Path(a.nz_dir) / f"{T}_z{ib}_nz.txt", unpack=True)
            b0, al = BIAS_FALLBACK.get(T, (2.0, 0.6))
            b = biasmap.get((T, ib), b0 * (1 + 0.5 * (z.mean())) ** al)
            ell, gT, _ = camb_window_cls(cosmo, z, nz, b, a.lmax)
            t = bandaverage(ell, gT, lo, hi)
            C = bin_block(T, ib)
            Cinv = (nsims - len(d) - 2) / (nsims - 1) * np.linalg.inv(C)
            tCt = float(t @ Cinv @ t)
            A[j] = float(t @ Cinv @ d) / tCt
        return A

    def joint(A):
        m = np.isfinite(A)
        S = Sigma[np.ix_(np.where(m)[0], np.where(m)[0])]
        Sinv = (nsims - m.sum() - 2) / (nsims - 1) * np.linalg.inv(S)
        one = np.ones(m.sum()); den = float(one @ Sinv @ one)
        return float((one @ Sinv @ A[m]) / den), float(1 / np.sqrt(den))

    import csv
    rows = []
    print("\nFiducial-cosmology robustness of A_ISW (bias & n(z) held fixed)")
    print(f"  {'cosmology':12s} {'joint A':>9s} {'σ_A':>7s}  Δ(joint A) vs fiducial")
    A_fid = A_perbin(cp); Af, sAf = joint(A_fid)
    for name, cosmo in variants(cp).items():
        A = A_fid if name == "fiducial" else A_perbin(cosmo)
        Aj, sAj = joint(A)
        dshift = Aj - Af
        print(f"  {name:12s} {Aj:>9.3f} {sAj:>7.3f}  {dshift:>+8.3f}  ({dshift/sAf:>+.2f} σ_A)")
        rows.append(dict(cosmology=name, joint_A=round(Aj, 4), sigma_A=round(sAj, 4),
                         dA_vs_fid=round(dshift, 4), dA_over_sigma=round(dshift / sAf, 3)))

    spread = max(r["joint_A"] for r in rows) - min(r["joint_A"] for r in rows)
    print(f"\n  full spread in joint A across ±1σ cosmologies: {spread:.3f} "
          f"= {spread/sAf:.2f} σ_A  -> {'robust' if spread < 0.5*sAf else 'CHECK'}")
    with open(a.out_csv, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote {a.out_csv}\n")


if __name__ == "__main__":
    main()
