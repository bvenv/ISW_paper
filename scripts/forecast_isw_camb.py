#!/usr/bin/env python3
"""
Honest ISW S/N forecast for DESI×Planck using CAMB source windows.

Replaces the old forecast_isw_constraints.py numbers, which were ~10x too optimistic
because they used the broken clank ISW templates. Here gT_i = TxW_i (ISW×galaxy) and
gg_ij = W_i×W_j come from one CAMB source-window run (full Boltzmann), with real shot
noise N_i = 1/nbar_i and per-tracer sky areas from the DESI n(z) tables.

For each tracer (one redshift bin = full n(z)):
    F_t = Σ_l (2l+1) fsky_t * gT_t² / [ (gg_tt + N_t) C_l^TT + gT_t² ]
    sigma(A)_t = 1/sqrt(F_t),  S/N_t = sqrt(F_t)   (if A_ISW=1)

Joint (all tracers, common footprint, full cross-covariance):
    Cov_ij(l) = [ (gg_ij + N_i delta_ij) C_l^TT + gT_i gT_j ] / [(2l+1) fsky_joint]
    F = Σ_l gT(l)^T Cov(l)^-1 gT(l)
"""
import argparse
import numpy as np
import camb
from camb.sources import SplinedSourceWindow

DEG2_FULLSKY = 41252.96
TRACERS = ["BGS", "LRG", "ELG", "QSO"]
BIAS = {"BGS": 1.34, "LRG": 2.05, "ELG": 1.3, "QSO": 2.3}  # effective linear bias
_COUNTS_OFF = ("counts_redshift", "counts_lensing", "counts_velocity", "counts_radial",
               "counts_timedelay", "counts_ISW", "counts_potential", "counts_evolve")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--nz-dir", default=".", help="Dir with {TRACER}_NGCplusSGC_nz.txt")
    p.add_argument("--lmin", type=int, default=2)
    p.add_argument("--lmax", type=int, default=100)
    p.add_argument("--fsky-joint", type=float, default=0.15,
                   help="Common footprint for the joint (correlated) forecast")
    p.add_argument("--h", type=float, default=0.6736)
    p.add_argument("--ombh2", type=float, default=0.02237)
    p.add_argument("--omch2", type=float, default=0.1200)
    p.add_argument("--ns", type=float, default=0.9649)
    p.add_argument("--As", type=float, default=2.1e-9)
    return p.parse_args()


def load_nz(path):
    a = np.loadtxt(path)
    zmid, Ntot = a[:, 0], a[:, 4]
    nz = a[:, 3]
    area = [float(l.split(":")[1]) for l in open(path) if "effective area" in l][0]
    nbar_sr = Ntot.sum() / (area * (np.pi / 180) ** 2)
    fsky = area / DEG2_FULLSKY
    return zmid, np.clip(nz, 0, None), nbar_sr, fsky


def main():
    args = parse_args()
    cp = dict(H0=100 * args.h, ombh2=args.ombh2, omch2=args.omch2, ns=args.ns, As=args.As, tau=0.054)

    # one CAMB run with all four windows -> TxWi (gT) and WixWj (gg)
    pars = camb.set_params(**cp)
    windows, nbar, fsky_t = [], {}, {}
    for t in TRACERS:
        z, nz, nb, fs = load_nz(f"{args.nz_dir}/{t}_NGCplusSGC_nz.txt")
        nz = nz / np.trapz(nz, z)
        windows.append(SplinedSourceWindow(z=z, W=nz, source_type="counts", bias=BIAS[t]))
        nbar[t], fsky_t[t] = nb, fs
    pars.SourceWindows = windows
    pars.SourceTerms.counts_density = True
    for term in _COUNTS_OFF:
        setattr(pars.SourceTerms, term, False)
    pars.SourceTerms.limber_windows = False
    pars.set_for_lmax(args.lmax + 80)
    res = camb.get_results(pars)
    d = res.get_cmb_unlensed_scalar_array_dict(CMB_unit="muK", raw_cl=True)
    ctt = res.get_cmb_power_spectra(pars, lmax=args.lmax + 80, raw_cl=True, CMB_unit="muK")["total"][:, 0]

    ell = np.arange(args.lmin, args.lmax + 1)
    n = len(TRACERS)
    gT = np.array([d[f"TxW{i+1}"][ell] for i in range(n)])               # (n, nl) µK
    gg = np.array([[d[f"W{i+1}xW{j+1}"][ell] for j in range(n)] for i in range(n)])  # (n,n,nl)
    TT = ctt[ell]
    Nsh = np.array([1.0 / nbar[t] for t in TRACERS])

    # ---- per-tracer single-bin Fisher (own fsky) ----
    print("Per-tracer ISW forecast (single bin = full n(z)):")
    print(f"  {'tracer':6s} {'zmean':>6s} {'fsky':>6s} {'sigma(A)':>9s} {'S/N(A=1)':>9s}")
    SN = {}
    for i, t in enumerate(TRACERS):
        var = (gg[i, i] + Nsh[i]) * TT + gT[i] ** 2
        F = np.sum((2 * ell + 1) * fsky_t[t] * gT[i] ** 2 / var)
        SN[t] = np.sqrt(F)
        zmean = np.average(load_nz(f'{args.nz_dir}/{t}_NGCplusSGC_nz.txt')[0],
                           weights=load_nz(f'{args.nz_dir}/{t}_NGCplusSGC_nz.txt')[1])
        print(f"  {t:6s} {zmean:6.2f} {fsky_t[t]:6.3f} {1/np.sqrt(F):9.3f} {np.sqrt(F):9.2f}")

    naive = np.sqrt(sum(s ** 2 for s in SN.values()))
    print(f"\n  naive independent combine (upper bound):  S/N = {naive:.2f}")

    # ---- joint Fisher, common footprint, full cross-covariance ----
    F = 0.0
    for li, l in enumerate(ell):
        C = ((gg[:, :, li] + np.diag(Nsh)) * TT[li]
             + np.outer(gT[:, li], gT[:, li])) / ((2 * l + 1) * args.fsky_joint)
        g = gT[:, li]
        F += g @ np.linalg.solve(C, g)
    print(f"  joint Fisher (fsky={args.fsky_joint}, full cross-cov): "
          f"sigma(A)={1/np.sqrt(F):.3f}  S/N={np.sqrt(F):.2f}")
    print("\n(ISW is cosmic-variance limited at low ell; S/N is robust to the bias choice.)")


if __name__ == "__main__":
    main()
