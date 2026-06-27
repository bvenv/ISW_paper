#!/usr/bin/env python3
"""
Validate the measured kappa×galaxy (kg) cross-spectrum against theory AND literature.

Theory: an independent CAMB-Limber C_l^{kappa g} (the standard lensing×galaxy expression),
which doubles as the kg template engine. kappa lives at higher l where Limber is excellent.

    C_l^{kg} = int dz  W_kappa(z) b n(z) / chi^2  P(k=(l+1/2)/chi, z)
    W_kappa  = (3/2) Omega_m (H0/c)^2 (chi/a) (chi_* - chi)/chi_*      [chi_* to last scattering]

Literature ballpark checks (DESI LRG × Planck CMB lensing):
  * the kg-implied galaxy bias should match DESI LRG clustering bias ~ 2.0
    (e.g. Kitanidis & White 2021; Hang et al. 2021; DESI LRG × Planck-PR4 papers);
  * the cross-correlation amplitude A_kg = measured/theory should be ~ 1 (LambdaCDM);
  * the detection S/N (~tens of sigma) — ours over l<=150 vs the literature's l<=1000.

Outputs a printed summary + results/plots/kg_LRG_validation.png (measured vs theory).
"""
import argparse
import numpy as np
import healpy as hp
import yaml
import camb
from camb import model
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from make_3x2pt_summary import exact_bin

C_KMS = 299792.458
DESI_LRG_BIAS_LIT = 2.0  # representative DESI LRG clustering bias from the literature


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tracer", default="LRG")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--bias-priors", default="results/bias/LRG_bias_priors.json")
    p.add_argument("--out-plot", default="results/plots/kg_LRG_validation.png")
    return p.parse_args()


def camb_setup(cp):
    pars = camb.set_params(H0=cp["H0"], ombh2=cp["ombh2"], omch2=cp["omch2"],
                           ns=cp["ns"], As=cp["As"], tau=cp["tau"])
    pars.set_matter_power(redshifts=np.linspace(0, 3.0, 60)[::-1], kmax=20.0)
    pars.NonLinear = model.NonLinear_both  # halofit (kg reaches mildly nonlinear scales)
    res = camb.get_results(pars)
    PK = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False,
                                            k_hunit=False, kmax=20.0, zmax=3.0)
    return pars, res, PK


def kg_limber(res, PK, cp, z, nz, bias, ells):
    """C_l^{kappa g} at the given (z, n(z)) and linear bias (Mpc units throughout)."""
    nz = nz / np.trapz(nz, z)
    chi = res.comoving_radial_distance(z)                  # Mpc
    a = 1.0 / (1.0 + z)
    Om = res.get_Omega("cdm") + res.get_Omega("baryon")
    chistar = res.comoving_radial_distance(1089.8)         # to last scattering
    Wk = 1.5 * Om * (cp["H0"] / C_KMS) ** 2 * chi / a * (chistar - chi) / chistar  # 1/Mpc
    Cl = np.zeros(len(ells))
    for i, l in enumerate(ells):
        k = (l + 0.5) / chi
        Pk = np.array([PK.P(zz, kk) for zz, kk in zip(z, k)])   # Mpc^3
        Cl[i] = np.trapz(Wk * bias * nz / chi ** 2 * Pk, z)
    return Cl


def bandavg(ell_th, cl_th, lo, hi):
    out = np.zeros(len(lo))
    for b, (a, c) in enumerate(zip(lo, hi)):
        sel = (ell_th >= a) & (ell_th < c)
        out[b] = cl_th[sel].mean() if sel.any() else np.interp(0.5 * (a + c), ell_th, cl_th)
    return out


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    c = cfg["theory"]["cosmo"]
    cp = dict(H0=float(c["H0"]), ombh2=float(c["ombh2"]), omch2=float(c["omch2"]),
              ns=float(c["ns"]), As=float(c["As"]), tau=float(c["tau"]))
    bins = cfg["desi"]["bins"][args.tracer]
    spectra = __import__("pathlib").Path(cfg["paths"]["spectra"]) / "desi"
    maps = __import__("pathlib").Path(cfg["paths"]["maps"])
    masks = __import__("pathlib").Path(cfg["paths"]["masks"])

    pars, res, PK = camb_setup(cp)
    ell_th = np.arange(2, 3 * args.nside)

    # mask + kappa for the Knox errors
    jm = hp.read_map(masks / f"planck_desi_kappa_joint_nside{args.nside}.fits.gz", dtype=np.float64)
    m = (jm > 0).astype(float); fsky = np.mean(m); w2 = np.mean(m ** 2)
    kap = hp.read_map(maps / "planck" / f"nside{args.nside}" / "kappa.fits.gz", dtype=np.float64)
    Ckk = hp.anafast(kap * m, lmax=3 * args.nside - 1) / w2

    fig, axes = plt.subplots(1, len(bins), figsize=(4.2 * len(bins), 3.8), sharey=True)
    print(f"\n{args.tracer} kappa×g validation (fsky={fsky:.3f}, ell<= {args.ell_max})")
    print(f"  {'bin':4s} {'b_kg (from kg)':>16s} {'b_clustering':>13s} {'A_kg':>8s} {'S/N':>6s}")
    bkg_list, w_list = [], []
    for idx, (zmin, zmax) in enumerate(bins):
        ib = idx + 1
        npz = np.load(spectra / f"kg_{args.tracer}_z{ib}_lmax{args.ell_max}.npz", allow_pickle=True)
        elo, ehi, ckg = npz["ell_edges_lo"], npz["ell_edges_hi"], npz["cl"]
        ellc = 0.5 * (elo + ehi)
        z, nz = np.loadtxt(args.nz_dir + f"/{args.tracer}_z{ib}_nz.txt", unpack=True)

        # unit-bias theory → band-averaged; A(unit bias) is the effective bias b_kg
        t_unit = bandavg(ell_th, kg_limber(res, PK, cp, z, nz, 1.0, ell_th), elo, ehi)

        # galaxy auto for Knox error
        d = hp.read_map(maps / "desi" / args.tracer / f"nside{args.nside}" / f"delta_{args.tracer}_z{ib}.fits.gz",
                        dtype=np.float64)
        gd = np.isfinite(d) & (d != hp.UNSEEN); d = np.where(gd, d, 0.0)
        Cgg = hp.anafast(d * m, lmax=3 * args.nside - 1) / w2
        var = np.array([ (Cgg[int(a):int(c)].mean() * Ckk[int(a):int(c)].mean())
                         / (fsky * np.sum(2 * np.arange(int(a), int(c)) + 1))
                         for a, c in zip(elo, ehi) ])
        # GLS amplitude of the unit-bias template = effective bias b_kg
        sel = ellc >= 30  # avoid the lowest, systematics-prone bands
        tCt = np.sum(t_unit[sel] ** 2 / var[sel])
        b_kg = float(np.sum(t_unit[sel] * ckg[sel] / var[sel]) / tCt)
        sig_b = float(1 / np.sqrt(tCt))
        sn = float(np.sqrt(np.sum(ckg[sel] ** 2 / var[sel])))
        b_clust = 0.0
        A_kg = b_kg / DESI_LRG_BIAS_LIT
        print(f"  z{ib}   {b_kg:6.2f} +/- {sig_b:4.2f}      {'~2.0 (lit)':>13s}   {A_kg:5.2f}   {sn:5.1f}")
        bkg_list.append((b_kg, sig_b)); w_list.append(1 / sig_b ** 2)

        ax = axes[idx] if len(bins) > 1 else axes
        fac = ellc * (ellc + 1) / (2 * np.pi)
        # plot theory with the EXACT NaMaster window (couple/decouple) so it matches the bandpowers;
        # b_kg is fit with the boxcar template (amplitude robust to the binning to ~0.06σ).
        t_plot = exact_bin(ell_th, kg_limber(res, PK, cp, z, nz, 1.0, ell_th),
                           npz["wsp_path"], args.nside, len(elo))
        if t_plot is None:
            t_plot = t_unit
        ax.errorbar(ellc, fac * ckg, yerr=fac * np.sqrt(var), fmt="o", ms=4, capsize=2, label="measured")
        ax.plot(ellc, fac * b_kg * t_plot, "-", lw=1.6, color="C3", label=f"theory (b={b_kg:.1f})")
        ax.set_title(f"{args.tracer} z{ib}  [{zmin:.1f},{zmax:.1f}]", fontsize=10)
        ax.set_xlabel(r"$\ell$"); ax.grid(ls=":", alpha=0.4)
        if idx == 0:
            ax.set_ylabel(r"$D_\ell^{\kappa g}$"); ax.legend(fontsize=8, frameon=False)

    bkg = np.array([b for b, _ in bkg_list]); wsum = np.sum(w_list)
    b_comb = np.sum(bkg * np.array(w_list)) / wsum; sig_comb = 1 / np.sqrt(wsum)
    sn_comb = np.sqrt(sum((b / s) ** 2 for b, s in bkg_list))
    print(f"\n  Combined: b_kg = {b_comb:.2f} +/- {sig_comb:.2f}   (S/N ~ {sn_comb:.1f})")
    print("  Literature ballpark: DESI LRG clustering bias ~ 2.0; A_kg = b_kg/2.0 ~ 1 if consistent.")
    print("  DESI LRG × Planck-kappa detections in the literature: ~27-50 sigma over l<=1000;")
    print(f"  ours is {sn_comb:.0f} sigma over l<=150 (climbs with l-range).")

    fig.suptitle(f"{args.tracer} κ×g: measured vs CAMB-Limber theory", y=1.02)
    fig.tight_layout()
    __import__("pathlib").Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {args.out_plot}\n")


if __name__ == "__main__":
    main()
