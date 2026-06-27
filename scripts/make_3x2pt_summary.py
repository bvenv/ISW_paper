#!/usr/bin/env python3
"""
3x2pt-style capstone summary for one tracer: gg (clustering), kg (CMB lensing x galaxy),
gT (ISW x galaxy) together — measured bandpowers vs theory, per tomographic bin, plus a
summary table (kappa-pinned bias, kg detection S/N, A_ISW).

Reuses the measurement + theory machinery from fit_bias_growth / build_isw_templates.
Outputs results/plots/{TRACER}_3x2pt.png and results/{TRACER}_3x2pt_summary.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import healpy as hp
import yaml
import pymaster as nmt
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import compute_crosscls as cc
from fit_bias_growth import camb_setup, theory_unitbias, measure_gg, bandavg

_WSP = {}


def _load_wsp(path):
    if path not in _WSP:
        w = nmt.NmtWorkspace(); w.read_from(str(path)); _WSP[path] = w
    return _WSP[path]


def exact_bin(ell_in, cl_in, wsp_path, nside, n_keep):
    """Bin theory with the EXACT NaMaster window: C_b = decouple(couple(C_l)). Falls back to the
    (2l+1) boxcar `bandavg` if the workspace file is unavailable."""
    if wsp_path is None:
        return None
    wsp_path = str(wsp_path)
    if not Path(wsp_path).exists():
        return None
    cl_full = np.zeros(3 * nside)
    ell_in = np.asarray(ell_in, int); cl_in = np.asarray(cl_in, float)
    sel = ell_in < 3 * nside
    cl_full[ell_in[sel]] = cl_in[sel]
    wsp = _load_wsp(wsp_path)
    return wsp.decouple_cell(wsp.couple_cell(np.array([cl_full])))[0][:n_keep]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--tracer", default="LRG")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--delta-ell", type=int, default=10)
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--bias-json", default=None,
                   help="default: results/bias/{tracer}_bias_magcorr.json (else _bias_kappa.json)")
    p.add_argument("--aisw-csv", default=None,
                   help="default: results/desi_Aisw_{tracer}_simcov.csv")
    p.add_argument("--boxcar", action="store_true",
                   help="use the (2l+1) boxcar theory binning instead of the exact NaMaster window")
    return p.parse_args()


def Dl(ell, cl):
    return ell * (ell + 1) / (2 * np.pi) * cl


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    c = cfg["theory"]["cosmo"]
    cp = dict(H0=float(c["H0"]), ombh2=float(c["ombh2"]), omch2=float(c["omch2"]),
              ns=float(c["ns"]), As=float(c["As"]), tau=float(c["tau"]))
    bins = cfg["desi"]["bins"][a.tracer]
    nb = len(bins)
    spectra = Path(cfg["paths"]["spectra"]) / "desi"
    maps = Path(cfg["paths"]["maps"])
    masks = Path(cfg["paths"]["masks"])

    bias_json = a.bias_json
    if bias_json is None:
        magcorr = f"results/bias/{a.tracer}_bias_magcorr.json"
        bias_json = magcorr if Path(magcorr).exists() else f"results/bias/{a.tracer}_bias_kappa.json"
    a.aisw_csv = a.aisw_csv or f"results/desi_Aisw_{a.tracer}_simcov.csv"
    bias = {p["ibin"]: p for p in json.load(open(bias_json))["priors"]}
    aisw = {}
    for r in csv.DictReader(open(a.aisw_csv)):
        try:
            aisw[int(r["ibin"])] = (float(r["A"]), float(r["sigmaA"]))
        except (ValueError, KeyError):
            pass

    res, PK, _ = camb_setup(cp)
    ell_th = np.arange(2, 3 * a.nside)
    binning = cc._make_bin(a.nside, 2, a.ell_max, a.delta_ell)
    pw = hp.pixwin(a.nside, lmax=3 * a.nside - 1)

    # kappa auto for kg errors
    kmask = masks / f"planck_desi_kappa_joint_nside{a.nside}.fits.gz"
    mk = (hp.read_map(kmask, dtype=np.float64) > 0).astype(float)
    fsky, w2 = np.mean(mk), np.mean(mk ** 2)
    kap = hp.read_map(maps / "planck" / f"nside{a.nside}" / "kappa.fits.gz", dtype=np.float64)
    Ckk = hp.anafast(kap * mk, lmax=3 * a.nside - 1) / w2

    fig, axes = plt.subplots(nb, 3, figsize=(11, 2.7 * nb), squeeze=False)
    col = ["gg  (clustering)", r"$\kappa$g  (CMB lensing)", "gT  (ISW)"]
    rows = []
    for idx, (zmin, zmax) in enumerate(bins):
        ib = idx + 1
        b = bias[ib]["b"]
        z, nz = np.loadtxt(a.nz_dir + f"/{a.tracer}_z{ib}_nz.txt", unpack=True)

        # measured
        ell_g, cl_gg, _ = measure_gg(cfg, a.tracer, ib, a.nside, kmask, binning, 1.0, "C2", "healpy-gauss", 2, a.ell_max)
        kg = np.load(spectra / f"kg_{a.tracer}_z{ib}_lmax{a.ell_max}.npz", allow_pickle=True)
        gt = np.load(spectra / f"gT_{a.tracer}_z{ib}_lmax{a.ell_max}.npz", allow_pickle=True)
        elo, ehi = kg["ell_edges_lo"], kg["ell_edges_hi"]; ellc = 0.5 * (elo + ehi)
        pwb = bandavg(np.arange(len(pw)), pw, elo, ehi)
        cl_gg = cl_gg / pwb ** 2; cl_kg = kg["cl"]; cl_gT = gt["cl"]  # kg/gT deconvolved at source

        # theory (b from kappa-pinned fit; gT template already at that b).
        # Bin with the EXACT NaMaster window (couple/decouple) so the plotted theory matches the
        # data bandpowers; fall back to the (2l+1) boxcar if a workspace is missing or --boxcar.
        gg_unit_l = theory_unitbias(res, PK, cp, z, nz, ell_th, "gg")
        kg_unit_l = theory_unitbias(res, PK, cp, z, nz, ell_th, "kg")
        gTt = np.load("templates/isw/gT_%s_z%d.npz" % (a.tracer, ib))
        nk = len(elo)
        wsp_gg = spectra / "workspaces" / f"wsp_auto_{a.tracer}_z{ib}_n{a.nside}_d{a.delta_ell}_apo1.00L.fits"
        Tgg_unit = None if a.boxcar else exact_bin(ell_th, gg_unit_l, wsp_gg, a.nside, nk)
        Tkg = None if a.boxcar else exact_bin(ell_th, kg_unit_l, kg["wsp_path"], a.nside, nk)
        TgT = None if a.boxcar else exact_bin(gTt["ell"], gTt["cl"], gt["wsp_path"], a.nside, nk)
        if Tgg_unit is None:
            Tgg_unit = bandavg(ell_th, gg_unit_l, elo, ehi)
        Tkg = b * (Tkg if Tkg is not None else bandavg(ell_th, kg_unit_l, elo, ehi))
        if TgT is None:
            TgT = bandavg(gTt["ell"], gTt["cl"], elo, ehi)

        # errors (Knox / sims)
        d = cc._load_map_ring(str(maps / "desi" / a.tracer / f"nside{a.nside}" / f"delta_{a.tracer}_z{ib}.fits.gz"), a.nside)
        gd = np.isfinite(d) & (d != hp.UNSEEN)
        Cgg_l = hp.anafast(np.where(gd, d, 0.0) * mk, lmax=3 * a.nside - 1) / w2
        nmodes = np.array([fsky * np.sum(2 * np.arange(int(x), int(y)) + 1) for x, y in zip(elo, ehi)])
        Cgg_b = np.array([Cgg_l[int(x):int(y)].mean() for x, y in zip(elo, ehi)])
        Ckk_b = np.array([Ckk[int(x):int(y)].mean() for x, y in zip(elo, ehi)])
        e_gg = np.sqrt(2 * Cgg_b ** 2 / nmodes); e_kg = np.sqrt((Cgg_b * Ckk_b + cl_kg ** 2) / nmodes)

        # gg model = clustering (kappa-pinned b^2) + fitted shot noise N (so it matches the data)
        s = ellc >= 30; w = 1 / e_gg[s] ** 2; T = Tgg_unit[s]; dd = cl_gg[s]
        det = np.sum(w * T * T) * np.sum(w) - np.sum(w * T) ** 2
        Nsh = (np.sum(w * T * T) * np.sum(w * dd) - np.sum(w * T) * np.sum(w * dd * T)) / det
        Tgg = b ** 2 * Tgg_unit + Nsh
        try:
            sc = np.load(f"results/sim_cov_{a.tracer}.npz"); nbd = int(sc["nband"])
            blk = sc["cov"][(ib - 1) * nbd:ib * nbd, (ib - 1) * nbd:ib * nbd]
            e_gT = np.sqrt(np.diag(blk))
        except Exception:
            e_gT = np.full_like(cl_gT, np.nan)

        for j, (m, t, e) in enumerate([(cl_gg, Tgg, e_gg), (cl_kg, Tkg, e_kg), (cl_gT, TgT, e_gT)]):
            ax = axes[idx][j]
            ax.errorbar(ellc, Dl(ellc, m), yerr=Dl(ellc, e), fmt="o", ms=3.5, capsize=2, lw=1)
            ax.plot(ellc, Dl(ellc, t), "-", color="C3", lw=1.6)
            ax.axhline(0, ls=":", lw=.7, c="gray"); ax.grid(ls=":", alpha=.3)
            if idx == 0: ax.set_title(col[j], fontsize=11)
            if j == 0: ax.set_ylabel(f"z{ib} [{zmin:.1f},{zmax:.1f}]\n" + r"$D_\ell$", fontsize=9)
            if idx == nb - 1: ax.set_xlabel(r"$\ell$")

        A, sA = aisw.get(ib, (np.nan, np.nan))
        sn_kg = float(np.sqrt(np.sum((cl_kg / e_kg)[ellc >= 30] ** 2)))
        rows.append((f"z{ib}", f"{zmin:.1f}-{zmax:.1f}", round(b, 2), round(bias[ib]["sigma_b"], 2),
                     round(sn_kg, 1), round(A, 2), round(sA, 2)))

    fig.suptitle(f"{a.tracer} DESI×Planck 3×2pt:  gg + κg + gT   (measured vs theory)", y=1.0, fontsize=13)
    fig.tight_layout()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/plots/{a.tracer}_3x2pt.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    with open(f"results/{a.tracer}_3x2pt_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "z", "b(kappa-pinned)", "sigma_b", "kg_SN", "A_ISW", "sigmaA"])
        w.writerows(rows)

    print(f"\n{a.tracer} 3x2pt summary")
    print(f"  {'bin':4s} {'z':>9s} {'b':>6s} {'kg S/N':>7s} {'A_ISW':>13s}")
    for bn, zr, b, sb, sn, A, sA in rows:
        print(f"  {bn:4s} {zr:>9s} {b:6.2f} {sn:7.1f}   {A:5.2f} +/- {sA:.2f}")
    print(f"  -> results/plots/{a.tracer}_3x2pt.png + results/{a.tracer}_3x2pt_summary.csv\n")


if __name__ == "__main__":
    main()
