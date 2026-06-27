#!/usr/bin/env python3
"""
Joint multi-tracer gT bandpower covariance: one Gaussian CMB realisation crossed with EVERY
tracer×bin at once, so the sample covariance captures the cross-tracer correlations (all
tracers see the same CMB) that the per-tracer sim_covariance.py misses.

Outputs results/sim_cov_all.npz (full covariance over the stacked tracer×bin bandpowers, with
the (tracer, ibin) layout) and per-tracer results/desi_Aisw_{TRACER}_simcov.csv (per-bin A_ISW
from the diagonal blocks) for the plot/naive combine. combine_joint_aisw.py uses the full
covariance for the optimal joint amplitude.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import healpy as hp
import pymaster as nmt
import yaml

import compute_crosscls as cc
from fit_isw_amplitudes import bandaverage, load_theory_templates


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tracers", nargs="+", default=["LRG", "BGS", "ELG", "QSO"])
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--delta-ell", type=int, default=10)
    p.add_argument("--apod-deg", type=float, default=1.0)
    p.add_argument("--apotype", default="C2")
    p.add_argument("--apod-method", default="healpy-gauss")
    p.add_argument("--cmb-label", default="SMICA")
    p.add_argument("--nsims", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam-fwhm-arcmin", type=float, default=5.0)
    p.add_argument("--templates-dir", default="templates/isw")
    p.add_argument("--cltt", default="templates/isw/cltt_camb_uK2.txt")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    nside, L = a.nside, a.ell_max
    lmax_sht = 3 * nside - 1
    bins_cfg = cfg["desi"]["bins"]
    spectra = Path(cfg["paths"]["spectra"]) / "desi"

    # transfer for the sim CMB (beam + pixwin), matching the data T
    cltt = np.loadtxt(a.cltt); cl_tt = np.zeros(lmax_sht + 1)
    e = cltt[:, 0].astype(int); ok = e <= lmax_sht; cl_tt[e[ok]] = cltt[ok, 1]
    transfer = hp.gauss_beam(np.deg2rad(a.beam_fwhm_arcmin / 60.0), lmax=lmax_sht) * hp.pixwin(nside, lmax=lmax_sht)
    pwb_full = hp.pixwin(nside, lmax=lmax_sht)

    binning = cc._make_bin(nside, a.ell_min, L, a.delta_ell)
    ell_eff = binning.get_effective_ells()
    sel = (ell_eff >= max(2, a.ell_min)) & (ell_eff <= L)
    elo = np.array([binning.get_ell_min(i) for i in range(binning.get_n_bands())])[sel]
    ehi = np.array([binning.get_ell_max(i) for i in range(binning.get_n_bands())])[sel]
    nband = int(sel.sum())
    pwb = np.array([pwb_full[int(x):int(y) + 1].mean() for x, y in zip(elo, ehi)])

    # per tracer×bin: galaxy field, theory template, measured data; per tracer: mask + workspace
    layout = []          # (tracer, ibin)
    fg, dvec, tvec = [], [], []
    tr_mask, tr_wsp = {}, {}
    for T in a.tracers:
        for ib in range(1, len(bins_cfg[T]) + 1):
            dp, cp, jm, _ = cc._paths(cfg, nside, T, ib, a.cmb_label, "T")
            f_g, _fT, msk_apo, _ = cc._build_fields(str(dp), str(cp), cc._load_mask_ring(str(jm), nside),
                                                    a.apod_deg, a.apotype, a.apod_method)
            if T not in tr_mask:
                tr_mask[T] = msk_apo
                w = nmt.NmtWorkspace(); w.compute_coupling_matrix(f_g, f_g, binning); tr_wsp[T] = w
            layout.append((T, ib)); fg.append(f_g)
            data = np.load(spectra / f"gT_{T}_z{ib}_lmax{L}.npz", allow_pickle=True)
            dvec.append(data["cl"])
            ell_th, cl_gT_th, _g, _t = load_theory_templates(a.templates_dir, T, ib)
            tvec.append(bandaverage(ell_th, cl_gT_th, elo, ehi))
    nbin = len(layout); P = nbin * nband
    d_all = np.concatenate(dvec); t_all = np.concatenate(tvec)
    print(f"Joint covariance: {len(a.tracers)} tracers, {nbin} bins, {P} bandpowers, {a.nsims} sims")

    seeds = np.random.SeedSequence(a.seed).spawn(a.nsims)
    bp = np.zeros((a.nsims, P))
    for s in range(a.nsims):
        np.random.seed(int(seeds[s].generate_state(1)[0]))
        alm = hp.synalm(cl_tt, lmax=lmax_sht, new=True); hp.almxfl(alm, transfer, inplace=True)
        Tmap = hp.alm2map(alm, nside, lmax=lmax_sht, verbose=False)
        fT = {T: nmt.NmtField(tr_mask[T], [Tmap]) for T in a.tracers}
        for j, (T, ib) in enumerate(layout):
            cl = tr_wsp[T].decouple_cell(nmt.compute_coupled_cell(fg[j], fT[T]))[0][sel] / pwb
            bp[s, j * nband:(j + 1) * nband] = cl
        if (s + 1) % 50 == 0:
            print(f"  sim {s+1}/{a.nsims}")

    cov = np.cov(bp, rowvar=False)
    # Reduce to per-bin amplitudes (each bin's well-sampled nband×nband block), then build the
    # nbin×nbin AMPLITUDE covariance across sims. This captures cross-tracer correlations
    # (all bins share the sim CMB) while avoiding the unstable inversion of the full P×P (=165)
    # bandpower covariance at only nsims=300.
    hb = (a.nsims - nband - 2) / (a.nsims - 1)
    A_sim = np.zeros((a.nsims, nbin)); A_data = np.zeros(nbin); A_sig = np.zeros(nbin)
    for j in range(nbin):
        sl = slice(j * nband, (j + 1) * nband)
        cinv = hb * np.linalg.inv(cov[sl, sl]); t = t_all[sl]
        tCt = float(t @ cinv @ t)
        A_sim[:, j] = (bp[:, sl] @ cinv @ t) / tCt          # per-bin amplitude, each sim
        A_data[j] = float(d_all[sl] @ cinv @ t / tCt)       # measured per-bin amplitude
        A_sig[j] = float(1 / np.sqrt(tCt))                  # per-bin error (diagonal)
    Sigma = np.cov(A_sim, rowvar=False)                     # amplitude covariance (cross-tracer)

    Path("results").mkdir(exist_ok=True)
    np.savez("results/sim_cov_all.npz", Sigma=Sigma, A_data=A_data, A_sig=A_sig,
             tracers=np.array([t for t, _ in layout]), ibins=np.array([i for _, i in layout]),
             nsims=a.nsims)
    per_tracer_rows = {T: [] for T in a.tracers}
    for j, (T, ib) in enumerate(layout):
        zmin, zmax = bins_cfg[T][ib - 1]
        per_tracer_rows[T].append((ib, zmin, zmax, A_data[j], A_sig[j]))
    for T, rows in per_tracer_rows.items():
        with open(f"results/desi_Aisw_{T}_simcov.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["tracer", "ibin", "zmin", "zmax", "A", "sigmaA", "cov"])
            for ib, zmn, zmx, A, sA in rows:
                w.writerow([T, ib, zmn, zmx, round(A, 6), round(sA, 6), "sim"])
    print("Wrote results/sim_cov_all.npz (amplitude cov) + per-tracer simcov CSVs")


if __name__ == "__main__":
    main()
