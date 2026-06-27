#!/usr/bin/env python3
"""
Stage A joint fit: {gg, kg} -> b(z) and sigma8, per tomographic bin.

C_l^{gg} = b^2 sigma8^2 (x clustering) + shot noise ;  C_l^{kg} = b sigma8^2 (x lensing).
With unit-bias, fiducial-sigma8 CAMB-Limber templates:
    A_gg = C^{gg}_clustering / T_gg(b=1) = b^2 (sigma8/sigma8_fid)^2   (shot noise fit out)
    A_kg = C^{kg}          / T_kg(b=1) = b   (sigma8/sigma8_fid)^2
  =>  b = A_gg / A_kg ,   sigma8 = sigma8_fid * A_kg / sqrt(A_gg)

gg is measured here (NaMaster auto on the kappa-joint mask, so gg and kg share the same
footprint); kg is read from compute_crosscls (--cmb-field kappa). Theory is the same
CAMB-Limber engine validated in validate_kappa_template.py.

Output: results/desi_bias_kappa.json  (b(z), sigma_b, sigma8 per bin) — the bias prior that
pins Stage B (ISW with b fixed) and a standalone growth (sigma8/S8) measurement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import healpy as hp
import pymaster as nmt
import yaml
import camb
from camb import model

import compute_crosscls as cc

C_KMS = 299792.458


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
    p.add_argument("--apodizer", default="healpy-gauss")
    p.add_argument("--fit-lmin", type=int, default=30,
                   help="Min ell in the amplitude fits (skip systematics-prone low ell)")
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--out", default="results/desi_bias_kappa.json")
    return p.parse_args()


def camb_setup(cp):
    pars = camb.set_params(H0=cp["H0"], ombh2=cp["ombh2"], omch2=cp["omch2"],
                           ns=cp["ns"], As=cp["As"], tau=cp["tau"])
    pars.set_matter_power(redshifts=np.linspace(0, 3.0, 60)[::-1], kmax=20.0)
    pars.NonLinear = model.NonLinear_both
    res = camb.get_results(pars)
    PK = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False,
                                            k_hunit=False, kmax=20.0, zmax=3.0)
    sigma8_fid = float(res.get_sigma8_0())
    return res, PK, sigma8_fid


def theory_unitbias(res, PK, cp, z, nz, ells, which):
    """Unit-bias, fiducial-sigma8 C_l for 'gg' or 'kg' (Mpc units)."""
    nz = nz / np.trapz(nz, z)
    chi = res.comoving_radial_distance(z)
    Hz = np.array([res.hubble_parameter(zz) for zz in z])
    a = 1.0 / (1.0 + z)
    Om = res.get_Omega("cdm") + res.get_Omega("baryon")
    chistar = res.comoving_radial_distance(1089.8)
    Wk = 1.5 * Om * (cp["H0"] / C_KMS) ** 2 * chi / a * (chistar - chi) / chistar
    Cl = np.zeros(len(ells))
    for i, l in enumerate(ells):
        Pk = np.array([PK.P(zz, (l + 0.5) / cc_chi) for zz, cc_chi in zip(z, chi)])
        if which == "gg":
            integ = nz ** 2 * (Hz / C_KMS) / chi ** 2 * Pk          # b^2 outside (=1)
        else:  # kg
            integ = Wk * nz / chi ** 2 * Pk                          # b outside (=1)
        Cl[i] = np.trapz(integ, z)
    return Cl


def bandavg(ell_th, cl_th, lo, hi):
    return np.array([cl_th[(ell_th >= a) & (ell_th < c)].mean()
                     if ((ell_th >= a) & (ell_th < c)).any()
                     else np.interp(0.5 * (a + c), ell_th, cl_th)
                     for a, c in zip(lo, hi)])


def measure_gg(cfg, tracer, ibin, nside, mask_path, binning, apod_deg, apotype, apodizer, lmin, lmax):
    """NaMaster galaxy auto on the given (kappa-joint) mask."""
    dpath = Path(cfg["paths"]["maps"]) / "desi" / tracer / f"nside{nside}" / f"delta_{tracer}_z{ibin}.fits.gz"
    delta = cc._load_map_ring(str(dpath), nside)
    jmask = cc._load_mask_ring(str(mask_path), nside)
    finite = np.isfinite(delta) & (delta != hp.UNSEEN)
    msk = jmask * finite
    msk_apo = cc._apodize(msk, apod_deg, apotype, method=apodizer)
    delta = np.where(finite, delta, 0.0)
    f_g = nmt.NmtField(msk_apo, [delta])
    wsp = nmt.NmtWorkspace(); wsp.compute_coupling_matrix(f_g, f_g, binning)
    cl = wsp.decouple_cell(nmt.compute_coupled_cell(f_g, f_g))[0]
    ell_eff = binning.get_effective_ells()
    sel = (ell_eff >= max(2, lmin)) & (ell_eff <= lmax)
    fsky2 = float(np.mean(msk_apo ** 2))
    return ell_eff[sel], cl[sel], fsky2


def fit_xN(T, d, var):
    """Weighted linear fit d = x*T + N; return x, sigma_x (x = clustering amplitude)."""
    w = 1.0 / var
    Stt = np.sum(w * T * T); St = np.sum(w * T); S1 = np.sum(w)
    Sdt = np.sum(w * d * T); Sd = np.sum(w * d)
    det = Stt * S1 - St * St
    x = (S1 * Sdt - St * Sd) / det
    var_x = S1 / det
    return float(x), float(np.sqrt(max(var_x, 0)))


def gls_amp(T, d, var):
    """GLS single-amplitude fit d = A*T; return A, sigma_A."""
    w = 1.0 / var
    tCt = np.sum(w * T * T)
    return float(np.sum(w * T * d) / tCt), float(1.0 / np.sqrt(tCt))


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    c = cfg["theory"]["cosmo"]
    cp = dict(H0=float(c["H0"]), ombh2=float(c["ombh2"]), omch2=float(c["omch2"]),
              ns=float(c["ns"]), As=float(c["As"]), tau=float(c["tau"]))
    bins = cfg["desi"]["bins"][args.tracer]
    spectra = Path(cfg["paths"]["spectra"]) / "desi"
    masks = Path(cfg["paths"]["masks"])
    kmask = masks / f"planck_desi_kappa_joint_nside{args.nside}.fits.gz"

    res, PK, sigma8_fid = camb_setup(cp)
    ell_th = np.arange(2, 3 * args.nside)
    binning = cc._make_bin(args.nside, args.ell_min, args.ell_max, args.delta_ell)

    # kappa auto (for the gg/kg Knox errors)
    kap = hp.read_map(Path(cfg["paths"]["maps"]) / "planck" / f"nside{args.nside}" / "kappa.fits.gz", dtype=np.float64)
    mk = (hp.read_map(kmask, dtype=np.float64) > 0).astype(float)
    fsky, w2 = np.mean(mk), np.mean(mk ** 2)
    Ckk = hp.anafast(kap * mk, lmax=3 * args.nside - 1) / w2

    print(f"\n{args.tracer} joint {{gg, kg}} fit  (sigma8_fid={sigma8_fid:.3f}, fsky={fsky:.3f})")
    print("  PRIMARY b(z) at sigma8=Planck (combine gg & kg);  *sigma8-free is noisy at DR1 S/N")
    print(f"  {'bin':4s} {'b (fixed s8)':>16s}")
    out = {"tracer": args.tracer, "sigma8_fid": sigma8_fid, "priors": []}

    for idx, (zmin, zmax) in enumerate(bins):
        ib = idx + 1
        z, nz = np.loadtxt(args.nz_dir + f"/{args.tracer}_z{ib}_nz.txt", unpack=True)

        # measured spectra
        ell_g, cl_gg, _ = measure_gg(cfg, args.tracer, ib, args.nside, kmask, binning,
                                     args.apod_deg, args.apotype, args.apodizer, args.ell_min, args.ell_max)
        kgz = np.load(spectra / f"kg_{args.tracer}_z{ib}_lmax{args.ell_max}.npz", allow_pickle=True)
        elo, ehi, cl_kg = kgz["ell_edges_lo"], kgz["ell_edges_hi"], kgz["cl"]
        ellc = 0.5 * (elo + ehi)

        # gg is measured here (δ×δ → W_pix^2); kg already has its δ pixel window deconvolved
        # in compute_crosscls, so only gg needs correcting.
        pw = hp.pixwin(args.nside, lmax=3 * args.nside - 1)
        pwb = bandavg(np.arange(len(pw)), pw, elo, ehi)
        cl_gg = cl_gg / pwb ** 2

        # unit-bias theory, band-averaged
        Tgg = bandavg(ell_th, theory_unitbias(res, PK, cp, z, nz, ell_th, "gg"), elo, ehi)
        Tkg = bandavg(ell_th, theory_unitbias(res, PK, cp, z, nz, ell_th, "kg"), elo, ehi)

        # Knox variances (measured autos)
        Cgg_l = hp.anafast((lambda d: np.where(np.isfinite(d) & (d != hp.UNSEEN), d, 0.0))(
            cc._load_map_ring(str(Path(cfg["paths"]["maps"]) / "desi" / args.tracer /
                                  f"nside{args.nside}" / f"delta_{args.tracer}_z{ib}.fits.gz"), args.nside)) * mk,
            lmax=3 * args.nside - 1) / w2
        def band_auto(C):
            return np.array([C[int(a):int(c)].mean() for a, c in zip(elo, ehi)])
        Cgg_b, Ckk_b = band_auto(Cgg_l), band_auto(Ckk)
        nmodes = np.array([fsky * np.sum(2 * np.arange(int(a), int(c)) + 1) for a, c in zip(elo, ehi)])
        var_gg = 2.0 * Cgg_b ** 2 / nmodes
        var_kg = (Cgg_b * Ckk_b + cl_kg ** 2) / nmodes

        sel = ellc >= args.fit_lmin
        A_gg, sA_gg = fit_xN(Tgg[sel], cl_gg[sel], var_gg[sel])      # b^2 s^2 (+N fit out)
        A_kg, sA_kg = gls_amp(Tkg[sel], cl_kg[sel], var_kg[sel])     # b s^2

        # PRIMARY: bias at fixed sigma8 = Planck fiducial (s=1). Both probes estimate b;
        # combine inverse-variance. This is the robust b(z) prior for Stage B.
        b_gg = float(np.sqrt(max(A_gg, 0.0))); sb_gg = float(0.5 * sA_gg / max(b_gg, 1e-9))
        b_kg = float(A_kg); sb_kg = float(sA_kg)
        wg, wk = 1 / sb_gg ** 2, 1 / sb_kg ** 2
        b = (wg * b_gg + wk * b_kg) / (wg + wk); sb = (wg + wk) ** -0.5

        # SECONDARY (noisy at DR1 S/N): free sigma8 from the gg/kg ratio
        sigma8 = sigma8_fid * A_kg / np.sqrt(A_gg)
        ssig = sigma8 * np.sqrt((sA_kg / A_kg) ** 2 + 0.25 * (sA_gg / A_gg) ** 2)
        print(f"  z{ib}   {b:5.2f} +/- {sb:4.2f}   (b_gg={b_gg:.2f} b_kg={b_kg:.2f})   "
              f"sigma8={sigma8:5.2f}+/-{ssig:4.2f}*")
        out["priors"].append({"tracer": args.tracer, "ibin": ib, "zmin": float(zmin), "zmax": float(zmax),
                              "b": float(b), "sigma_b": float(sb),
                              "b_gg": b_gg, "b_kg": b_kg,
                              "sigma8_free": float(sigma8), "sigma_sigma8_free": float(ssig),
                              "A_gg": float(A_gg), "A_kg": float(A_kg)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}  (b(z) prior for Stage B + sigma8 growth measurement)\n")


if __name__ == "__main__":
    main()
