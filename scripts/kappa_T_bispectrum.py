#!/usr/bin/env python3
"""
ISW–lensing cross-spectrum: Planck κ × Planck T (the CMB-internal ISW, no galaxies).

The same low-z potential Φ sources both the late ISW (in T) and the lensing convergence κ,
so ⟨κ T⟩ is non-zero and ISW-dominated at low ℓ. Since κ is reconstructed from the CMB
(κ ~ TT), ⟨κ T⟩ is effectively the ISW–lensing bispectrum ⟨TTT⟩. This is an independent
confirmation of the galaxy gT result — same low-z ISW, completely different tracer.

Measured on the DESI∩lensing footprint (same sky as our gT) via NaMaster; Knox errors from
the κ and T autos; theory C_ℓ^{κT} from CAMB (lensing-potential×T → convergence×T).
Outputs results/plots/kappaT_isw_lensing.png and a printed S/N.
"""
import argparse
from pathlib import Path

import numpy as np
import healpy as hp
import pymaster as nmt
import yaml
import camb
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import compute_crosscls as cc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--ell-min", type=int, default=8)
    p.add_argument("--ell-max", type=int, default=100)
    p.add_argument("--delta-ell", type=int, default=12)
    p.add_argument("--apod-deg", type=float, default=1.0)
    p.add_argument("--cmb-label", default="SMICA")
    p.add_argument("--out-plot", default="results/plots/kappaT_isw_lensing.png")
    return p.parse_args()


def field(m, mask):
    return nmt.NmtField(mask, [np.where(np.isfinite(m) & (m != hp.UNSEEN), m, 0.0)])


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    maps = Path(cfg["paths"]["maps"]) / "planck" / f"nside{a.nside}"
    masks = Path(cfg["paths"]["masks"])

    T = cc._load_map_ring(str(maps / f"cmb_T_{a.cmb_label}.fits.gz"), a.nside)
    K = cc._load_map_ring(str(maps / "kappa.fits.gz"), a.nside)
    jm = cc._load_mask_ring(str(masks / f"planck_desi_kappa_joint_nside{a.nside}.fits.gz"), a.nside)
    finite = np.isfinite(T) & (T != hp.UNSEEN)
    msk = cc._apodize(jm * finite, a.apod_deg, "C2", method="healpy-gauss")
    fsky = float(np.mean(msk > 0))

    binning = cc._make_bin(a.nside, a.ell_min, a.ell_max, a.delta_ell)
    fT, fK = field(T, msk), field(K, msk)
    wsp = nmt.NmtWorkspace(); wsp.compute_coupling_matrix(fK, fT, binning)
    dec = lambda f1, f2: wsp.decouple_cell(nmt.compute_coupled_cell(f1, f2))[0]
    cl_kT = dec(fK, fT); cl_kk = dec(fK, fK); cl_TT = dec(fT, fT)

    ell = binning.get_effective_ells()
    elo = np.array([binning.get_ell_min(i) for i in range(binning.get_n_bands())])
    ehi = np.array([binning.get_ell_max(i) for i in range(binning.get_n_bands())])
    sel = (ell >= a.ell_min) & (ell <= a.ell_max)
    nmodes = np.array([fsky * np.sum(2 * np.arange(int(x), int(y) + 1) + 1) for x, y in zip(elo, ehi)])
    var = (cl_kk * cl_TT + cl_kT ** 2) / nmodes
    sn = float(np.sqrt(np.sum((cl_kT[sel] ** 2 / var[sel]))))

    # theory C_l^{kT} from CAMB: lensing potential x T, converted to convergence x T
    c = cfg["theory"]["cosmo"]
    pars = camb.set_params(H0=float(c["H0"]), ombh2=float(c["ombh2"]), omch2=float(c["omch2"]),
                           ns=float(c["ns"]), As=float(c["As"]), tau=float(c["tau"]))
    pars.set_for_lmax(a.ell_max + 200, lens_potential_accuracy=2)
    res = camb.get_results(pars)
    d = res.get_cmb_unlensed_scalar_array_dict(CMB_unit="muK", raw_cl=True)
    Lth = np.arange(d["TxP"].size)
    clkT_th = 0.5 * Lth * (Lth + 1) * d["TxP"]            # phi -> kappa
    th_band = np.array([clkT_th[int(x):int(y) + 1].mean() if int(y) >= int(x) and int(y) < clkT_th.size
                        else 0.0 for x, y in zip(elo, ehi)])
    A = float(np.sum((th_band * cl_kT / var)[sel]) / np.sum((th_band ** 2 / var)[sel]))
    sA = float(1 / np.sqrt(np.sum((th_band ** 2 / var)[sel])))

    print(f"\nκ×T ISW–lensing cross (fsky={fsky:.3f}, ℓ={a.ell_min}-{a.ell_max})")
    print(f"  detection S/N (vs zero) = {sn:.1f}σ")
    print(f"  amplitude vs CAMB theory: A_κT = {A:.2f} ± {sA:.2f}  ({A/sA:.1f}σ)")
    print("  (CMB-internal ISW — independent of the galaxy gT, same low-z potential.)")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    fac = ell * (ell + 1) / (2 * np.pi)
    ax.errorbar(ell[sel], (fac * cl_kT)[sel], yerr=(fac * np.sqrt(var))[sel], fmt="o", capsize=3, label="measured κ×T")
    ax.plot(ell[sel], (fac * th_band)[sel], "-", color="C3", lw=1.6, label="CAMB ISW–lensing")
    ax.axhline(0, ls=":", c="gray"); ax.grid(ls=":", alpha=.4)
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{\kappa T}/2\pi\ \,[\mu K]$")
    ax.set_title(f"Planck κ × T (ISW–lensing), DESI footprint — {sn:.1f}σ")
    ax.legend(frameon=False)
    Path(a.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out_plot, dpi=150); plt.close(fig)
    print(f"  wrote {a.out_plot}\n")


if __name__ == "__main__":
    main()
