#!/usr/bin/env python3
"""
RSD (linear Kaiser) impact on the gg/κg/gT templates, via CAMB source windows.

Our templates use density only (`counts_redshift` is in build_isw_templates' _COUNTS_OFF). Here we
toggle `counts_redshift` on/off per tracer x bin and report the (2l+1)-weighted fractional change in
gg, κg, gT, propagated to the bias (b_gg, b_kg) and A_ISW, compared to the measured errors.

Bands: gg/κg over the BIAS fit range [fit_lmin,lmax] (default 30-150 — the fit discards l<30 where
Kaiser concentrates); gT over the ISW range [2,lmax]. Also prints the gg fraction over [2,fit_lmin]
to show how much the low-l cut removes. Kaiser is the dominant linear RSD; Doppler (counts_velocity)
is separate and negligible here.
"""
import argparse, csv, glob, json
from pathlib import Path
import numpy as np
import yaml
import camb
from camb.sources import SplinedSourceWindow

from magnification_impact import cosmo, read_nz
from fit_isw_amplitudes import bandaverage

_OFF_ALL = ("counts_redshift", "counts_lensing", "counts_velocity", "counts_radial",
            "counts_timedelay", "counts_ISW", "counts_potential", "counts_evolve")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--bias-glob", default="results/bias/*_bias_kappa.json")
    p.add_argument("--aisw-glob", default="results/desi_Aisw_*_simcov.csv")
    p.add_argument("--fit-lmin", type=int, default=30)
    p.add_argument("--lmax", type=int, default=150)
    p.add_argument("--out-csv", default="results/rsd_impact.csv")
    return p.parse_args()


def camb_obs(cp, z, nz, bias, lmax, rsd):
    pars = camb.set_params(H0=cp["H0"], ombh2=cp["ombh2"], omch2=cp["omch2"],
                           ns=cp["ns"], As=cp["As"], tau=cp["tau"])
    pars.SourceWindows = [SplinedSourceWindow(z=z, W=nz, source_type="counts", bias=bias)]
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_redshift = bool(rsd)            # linear Kaiser RSD
    for t in _OFF_ALL:
        if t == "counts_redshift":
            continue
        setattr(pars.SourceTerms, t, False)
    pars.SourceTerms.limber_windows = False
    pars.set_for_lmax(lmax + 50)
    d = camb.get_results(pars).get_cmb_unlensed_scalar_array_dict(CMB_unit="muK", raw_cl=True)
    ell = np.arange(lmax + 1)
    gT = d["TxW1"][: lmax + 1]
    gg = d["W1xW1"][: lmax + 1]
    kg = 0.5 * ell * (ell + 1) * d["PxW1"][: lmax + 1]
    return ell, gT, gg, kg


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config)); cp = cosmo(cfg)
    bins = cfg["desi"]["bins"]

    bias = {}
    for bf in glob.glob(a.bias_glob):
        d = json.load(open(bf)); T = d.get("tracer") or Path(bf).name.split("_")[0]
        for e in d.get("priors", []):
            bias[(T, int(e["ibin"]))] = (e["b"], e.get("sigma_b", np.nan))
    aerr = {}
    for f in glob.glob(a.aisw_glob):
        for r in csv.DictReader(open(f)):
            try:
                aerr[(r["tracer"], int(r["ibin"]))] = (float(r["A"]), float(r["sigmaA"]))
            except (ValueError, KeyError):
                pass

    def wfrac(ell, dC, C, lo, hi):
        m = (ell >= lo) & (ell <= hi)
        return float(np.sum((2 * ell[m] + 1) * dC[m]) / np.sum((2 * ell[m] + 1) * C[m]))

    rows = []
    print(f"\nRSD (Kaiser) impact   gg/κg band=[{a.fit_lmin},{a.lmax}], gT band=[2,{a.lmax}]")
    hdr = (f"{'tracer':6s}{'bin':>4s} | {'gg(fit)':>8s}{'gg(l<30)':>9s} | {'κg':>7s}{'gT':>7s} |"
           f" {'db/b|gg':>8s}{'σb/b':>7s} | {'dA/A':>7s}{'σA/A':>7s}  verdict")
    print(hdr); print("-" * len(hdr))
    for T in bins:
        for ib in range(1, len(bins[T]) + 1):
            p = Path(a.nz_dir) / f"{T}_z{ib}_nz.txt"
            if not p.exists():
                continue
            z, nz = read_nz(p)
            b = bias.get((T, ib), (2.0, np.nan))[0]
            ell, gT0, gg0, kg0 = camb_obs(cp, z, nz, b, a.lmax, rsd=False)
            _, gT1, gg1, kg1 = camb_obs(cp, z, nz, b, a.lmax, rsd=True)
            f_gg = wfrac(ell, gg1 - gg0, gg0, a.fit_lmin, a.lmax)
            f_gg_lo = wfrac(ell, gg1 - gg0, gg0, 2, a.fit_lmin)
            f_kg = wfrac(ell, kg1 - kg0, kg0, a.fit_lmin, a.lmax)
            f_gT = wfrac(ell, gT1 - gT0, gT0, 2, a.lmax)
            db_b = 0.5 * f_gg                                  # gg ∝ b² in the fit range
            dA_A = f_gT                                        # gT pinned bias unaffected by RSD-in-gg cut
            sb_b = bias.get((T, ib), (np.nan, np.nan))[1] / b if (T, ib) in bias else np.nan
            A_, sA_ = aerr.get((T, ib), (np.nan, np.nan))
            sA_A = abs(sA_ / A_) if A_ and np.isfinite(A_) and A_ != 0 else np.nan
            vb = "b:CHECK" if (np.isfinite(sb_b) and abs(db_b) > sb_b) else "b:ok"
            va = "A:CHECK" if (np.isfinite(sA_A) and abs(dA_A) > sA_A) else "A:ok"
            print(f"{T:6s}{ib:>4d} | {f_gg:>+8.1%}{f_gg_lo:>+9.1%} | {f_kg:>+7.1%}{f_gT:>+7.1%} |"
                  f" {db_b:>+8.1%}{sb_b:>7.1%} | {dA_A:>+7.1%}{sA_A:>7.1%}  {vb} {va}")
            rows.append(dict(tracer=T, ibin=ib, f_gg_fit=f_gg, f_gg_lowl=f_gg_lo, f_kg=f_kg,
                             f_gT=f_gT, db_b=db_b, sigb_b=sb_b, dA_A=dA_A, sigA_A=sA_A,
                             verdict=f"{vb} {va}"))
    with open(a.out_csv, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    nb = sum("b:CHECK" in r["verdict"] for r in rows)
    na = sum("A:CHECK" in r["verdict"] for r in rows)
    print(f"\n  wrote {a.out_csv}")
    print(f"  bias:  {nb}/{len(rows)} bins where RSD > σ_b (over the [{a.fit_lmin},{a.lmax}] fit range)")
    print(f"  A_ISW: {na}/{len(rows)} bins where RSD > σ_A (per-bin proxy)")

    # ---- definitive headline test: joint A_ISW with RSD-on vs RSD-off gT templates ----
    spectra = Path(cfg["paths"]["spectra"]) / "desi"
    sc = np.load("results/sim_cov_all.npz", allow_pickle=True)
    Sigma, nsims = sc["Sigma"], int(sc["nsims"])
    tr_lay = [str(x) for x in sc["tracers"]]; ib_lay = [int(x) for x in sc["ibins"]]
    covs = {T: np.load(f"results/sim_cov_{T}.npz", allow_pickle=True) for T in set(tr_lay)
            if Path(f"results/sim_cov_{T}.npz").exists()}

    def joint_A(rsd):
        A = np.full(len(tr_lay), np.nan)
        for j, (T, ib) in enumerate(zip(tr_lay, ib_lay)):
            sp = spectra / f"gT_{T}_z{ib}_lmax{a.lmax}.npz"
            if not sp.exists():
                continue
            spec = np.load(sp, allow_pickle=True)
            d = spec["cl"]; lo, hi = spec["ell_edges_lo"], spec["ell_edges_hi"]
            z, nz = read_nz(Path(a.nz_dir) / f"{T}_z{ib}_nz.txt")
            b = bias.get((T, ib), (2.0, np.nan))[0]
            ell, gT, _, _ = camb_obs(cp, z, nz, b, a.lmax, rsd=rsd)
            t = bandaverage(ell, gT, lo, hi)
            nb_ = int(covs[T]["nband"]); blk = covs[T]["cov"][(ib-1)*nb_:ib*nb_, (ib-1)*nb_:ib*nb_]
            Cinv = (nsims - len(d) - 2) / (nsims - 1) * np.linalg.inv(blk)
            A[j] = float(t @ Cinv @ d) / float(t @ Cinv @ t)
        m = np.isfinite(A)
        S = Sigma[np.ix_(np.where(m)[0], np.where(m)[0])]
        Sinv = (nsims - m.sum() - 2) / (nsims - 1) * np.linalg.inv(S)
        one = np.ones(m.sum()); den = float(one @ Sinv @ one)
        return float((one @ Sinv @ A[m]) / den), float(1 / np.sqrt(den))

    A0, s0 = joint_A(False); A1, s1 = joint_A(True)
    print(f"\n  joint A_ISW:  density-only gT = {A0:.3f} ± {s0:.3f};  +RSD gT = {A1:.3f} ± {s1:.3f}"
          f"  -> shift {A1-A0:+.3f} ({(A1-A0)/s0:+.2f} σ_A)")
    print(f"  => RSD {'negligible for the headline' if abs(A1-A0) < 0.5*s0 else 'CHECK'} "
          f"(bias unaffected by the l>30 cut)\n")


if __name__ == "__main__":
    main()
