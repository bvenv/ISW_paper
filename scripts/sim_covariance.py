#!/usr/bin/env python3
"""
Empirical (sims-based) bandpower covariance for the g×T ISW measurement, and a
covariance-correct refit of A_ISW.

The analytic diagonal Gaussian covariance in fit_isw_amplitudes.py is optimistic: it
ignores mask mode-coupling between bands and the bin-bin correlation induced by sharing
ONE Planck T map across all tomographic bins. Here we estimate the full covariance from
Monte-Carlo CMB realisations:

  for each sim:  draw a Gaussian T from C_l^TT (beam+pixwin), and cross the SAME T with
                 every galaxy bin's fixed delta_g field via NaMaster (same masks/workspace
                 as compute_crosscls). Stack the per-bin bandpowers.
  covariance  =  sample covariance of the stacked bandpower vector across sims.

Then refit A_ISW with this covariance (Hartlap-corrected inverse):
  per-bin  A_b   from the diagonal block,
  combined A     from the full covariance (bin-bin correlations included).

Outputs: results/sim_cov_{TRACER}.npz, results/desi_Aisw_{TRACER}_simcov.csv,
         results/plots/sim_corr_{TRACER}.png
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import healpy as hp
import pymaster as nmt
import yaml
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import compute_crosscls as cc
from fit_isw_amplitudes import bandaverage, load_theory_templates

LOGGER = logging.getLogger("sim_covariance")


def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tracer", default="LRG")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--delta-ell", type=int, default=10)
    p.add_argument("--apod-deg", type=float, default=1.0)
    p.add_argument("--apotype", choices=["C1", "C2"], default="C2")
    p.add_argument("--apod-method", choices=["healpy-gauss", "nmt", "none"], default="healpy-gauss")
    p.add_argument("--cmb-label", default="SMICA")
    p.add_argument("--nsims", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam-fwhm-arcmin", type=float, default=5.0,
                   help="Planck SMICA effective beam (~5'); applied with the pixel window")
    p.add_argument("--templates-dir", default="templates/isw")
    p.add_argument("--cltt", default="templates/isw/cltt_camb_uK2.txt")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = yaml.safe_load(open(args.config))
    nside, L = args.nside, args.ell_max
    bins = cfg["desi"]["bins"][args.tracer]
    nbins = len(bins)

    lmax_sht = 3 * nside - 1
    cltt = np.loadtxt(args.cltt)
    cl_tt = np.zeros(lmax_sht + 1)
    e = cltt[:, 0].astype(int); ok = e <= lmax_sht
    cl_tt[e[ok]] = cltt[ok, 1]
    bl = hp.gauss_beam(np.deg2rad(args.beam_fwhm_arcmin / 60.0), lmax=lmax_sht)
    pw = hp.pixwin(nside, lmax=lmax_sht)
    transfer = bl * pw

    # ---- fixed per-bin galaxy fields + workspaces (same setup as compute_crosscls) ----
    perbin = []
    masks_root = Path(cfg["paths"]["masks"])
    joint_mask = cc._load_mask_ring(str(masks_root / f"planck_desi_joint_nside{nside}.fits.gz"), nside)
    binning = cc._make_bin(nside, args.ell_min, L, args.delta_ell)
    ells_eff = binning.get_effective_ells()
    sel = (ells_eff >= max(2, args.ell_min)) & (ells_eff <= L)
    ell_lo = np.array([binning.get_ell_min(i) for i in range(binning.get_n_bands())])[sel]
    ell_hi = np.array([binning.get_ell_max(i) for i in range(binning.get_n_bands())])[sel]
    nband = int(sel.sum())

    for idx in range(nbins):
        ibin = idx + 1
        delta_p, cmb_p, jmask_p, _ = cc._paths(cfg, nside, args.tracer, ibin, args.cmb_label)
        f_g, _f_T, msk_apo, fsky = cc._build_fields(str(delta_p), str(cmb_p), joint_mask,
                                                    args.apod_deg, args.apotype, args.apod_method)
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(f_g, f_g, binning)  # mask-only coupling (same mask for g and T)
        perbin.append(dict(ibin=ibin, f_g=f_g, msk_apo=msk_apo, wsp=wsp, fsky=fsky))
        LOGGER.info("bin %d: fields built (fsky=%.3f)", ibin, fsky)

    # ---- Monte-Carlo: shared CMB crossed with every bin ----
    rng_seeds = np.random.SeedSequence(args.seed).spawn(args.nsims)
    bp = np.zeros((args.nsims, nbins, nband))  # [sim, bin, band]
    for s in range(args.nsims):
        seed_i = int(rng_seeds[s].generate_state(1)[0])
        np.random.seed(seed_i)
        alm = hp.synalm(cl_tt, lmax=lmax_sht, new=True)
        hp.almxfl(alm, transfer, inplace=True)
        Tmap = hp.alm2map(alm, nside, lmax=lmax_sht, verbose=False)
        for b, pb in enumerate(perbin):
            f_T = nmt.NmtField(pb["msk_apo"], [Tmap])
            cl_dec = pb["wsp"].decouple_cell(nmt.compute_coupled_cell(pb["f_g"], f_T))[0]
            bp[s, b] = cl_dec[sel]
        if (s + 1) % 50 == 0:
            LOGGER.info("  sim %d/%d", s + 1, args.nsims)

    X = bp.reshape(args.nsims, nbins * nband)            # stacked bandpowers
    cov = np.cov(X, rowvar=False)                        # (nbins*nband)^2
    p = cov.shape[0]
    hartlap = (args.nsims - p - 2) / (args.nsims - 1)    # unbiased inverse
    cinv = hartlap * np.linalg.inv(cov)

    # ---- data + templates, band-averaged onto the same bands ----
    spectra_root = Path(cfg["paths"]["spectra"]) / "desi"
    d_vec, t_vec, blocks = [], [], []
    for idx in range(nbins):
        ibin = idx + 1
        data = np.load(spectra_root / f"gT_{args.tracer}_z{ibin}_lmax{L}.npz", allow_pickle=True)
        d_vec.append(data["cl"])
        ell_th, cl_gT_th, _gg, _tt = load_theory_templates(args.templates_dir, args.tracer, ibin)
        t_vec.append(bandaverage(ell_th, cl_gT_th, ell_lo, ell_hi))
        blocks.append(slice(idx * nband, (idx + 1) * nband))
    d_vec = np.concatenate(d_vec); t_vec = np.concatenate(t_vec)

    # per-bin (diagonal block) and combined (full cov) GLS
    rows = []
    for idx in range(nbins):
        sl = blocks[idx]
        hb = (args.nsims - nband - 2) / (args.nsims - 1)   # Hartlap for the nband×nband block
        Cbb_inv = hb * np.linalg.inv(cov[sl, sl])
        t, d = t_vec[sl], d_vec[sl]
        tCt = t @ Cbb_inv @ t
        A = float((t @ Cbb_inv @ d) / tCt); sA = float(1.0 / np.sqrt(tCt))
        rows.append((idx + 1, bins[idx][0], bins[idx][1], A, sA))
        LOGGER.info("  %s z%d: A=%.3f ± %.3f (sim-cov)", args.tracer, idx + 1, A, sA)
    tCt_full = t_vec @ cinv @ t_vec
    A_all = float((t_vec @ cinv @ d_vec) / tCt_full); sA_all = float(1.0 / np.sqrt(tCt_full))
    LOGGER.info("  %s combined: A=%.3f ± %.3f  (%.1fσ from ΛCDM)", args.tracer, A_all, sA_all,
                (A_all - 1) / sA_all)

    # ---- save ----
    results = Path(cfg["paths"].get("results", "results")); results = Path("results")
    np.savez(results / f"sim_cov_{args.tracer}.npz", cov=cov, ell_lo=ell_lo, ell_hi=ell_hi,
             nbins=nbins, nband=nband, nsims=args.nsims, hartlap=hartlap,
             A_combined=A_all, sigmaA_combined=sA_all)
    with open(results / f"desi_Aisw_{args.tracer}_simcov.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tracer", "ibin", "zmin", "zmax", "A", "sigmaA", "cov"])
        for ib, zmn, zmx, A, sA in rows:
            w.writerow([args.tracer, ib, zmn, zmx, round(A, 6), round(sA, 6), "sim"])
        w.writerow([f"{args.tracer}_combined", "", "", "", round(A_all, 6), round(sA_all, 6), "sim"])

    # correlation matrix plot
    D = np.sqrt(np.diag(cov)); corr = cov / np.outer(D, D)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    for k in range(1, nbins):
        ax.axhline(k * nband - 0.5, color="k", lw=0.6); ax.axvline(k * nband - 0.5, color="k", lw=0.6)
    ax.set_title(f"{args.tracer} g×T bandpower correlation\n({nbins} bins × {nband} bands, {args.nsims} sims)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); (results / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(results / "plots" / f"sim_corr_{args.tracer}.png", dpi=150); plt.close(fig)
    LOGGER.info("Wrote sim covariance, simcov CSV, and correlation plot for %s", args.tracer)


if __name__ == "__main__":
    main()
