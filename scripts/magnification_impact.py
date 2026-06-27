#!/usr/bin/env python3
"""
Level-1 magnification-bias impact (CAMB source windows, full Boltzmann).

For each tracer x bin we run CAMB twice on the bin's n(z)+bias: (a) density only and
(b) density + magnification (counts_lensing on, slope s = dlog10Ndm), and extract the three
observables from one call:
    gT  = TxW1                      (ISW x galaxy)
    gg  = W1xW1                     (galaxy auto)
    kg  = 1/2 l(l+1) * PxW1         (CMB-lensing x galaxy; phi->kappa)
We report the (2l+1)-weighted fractional magnification contribution over the analysis band and
propagate it to the bias and A_ISW:
    db/b |_gg  ~ 1/2 * dCgg/Cgg ,  db/b |_kg ~ dCkg/Ckg ,  dA/A ~ dCgT/CgT - db/b|_kg
Then compare to the measured errors (sigma_b/b from the kappa-pinned bias, sigma_A/A) to decide
whether magnification matters per tracer. This QUANTIFIES the effect; it does not yet fold it in.

Magnification slopes s are representative literature values (EDIT `S_SLOPE` / `--slopes` and cite
the DESI magnification-slope reference before quoting in the paper).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import yaml
import camb
from camb.sources import SplinedSourceWindow

# Number-count slopes s = dlog10 N / dm per tracer (CAMB magnification prefactor is (5s-2)).
# Cited DESI values:
#   LRG  s~0.98  measured, Zhou+ 2023 DESI LRG x CMB-lensing (arXiv:2309.06443, Table 5: 0.97-1.04)
#   QSO  s=0.276 measured, DESI QSO fNL (arXiv:2305.07650)
#   BGS  s~0.88  from alpha_BGS=2.19 (alpha=2.5s convention, validated: alpha_LRG=2.52 -> s=1.008
#                vs measured 0.999); BGS magnification stays small from its low-z geometry
#   ELG  s~1.0   adopted (standard for DESI ELG; no clean per-bin measurement) -- FLAG in text
S_SLOPE = {"BGS": 0.88, "LRG": 0.98, "ELG": 1.00, "QSO": 0.28}

_OFF = ("counts_redshift", "counts_velocity", "counts_radial", "counts_timedelay",
        "counts_ISW", "counts_potential", "counts_evolve")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--bias-glob", default="results/bias/*_bias_kappa.json")
    p.add_argument("--aisw-glob", default="results/desi_Aisw_*_simcov.csv")
    p.add_argument("--lmin", type=int, default=2)
    p.add_argument("--lmax", type=int, default=150)
    p.add_argument("--slopes", default=None, help="override e.g. 'BGS:0.4,QSO:0.2'")
    p.add_argument("--out-csv", default="results/magnification_impact.csv")
    return p.parse_args()


def cosmo(cfg):
    c = cfg.get("theory", {}).get("cosmo", {})
    return dict(H0=float(c.get("H0", 67.36)), ombh2=float(c.get("ombh2", 0.02237)),
                omch2=float(c.get("omch2", 0.1200)), ns=float(c.get("ns", 0.9649)),
                As=float(c.get("As", 2.1e-9)), tau=float(c.get("tau", 0.054)))


def read_nz(path):
    a = np.loadtxt(path)
    z, nz = a[:, 0], (a[:, 3] if a.shape[1] >= 4 else a[:, 1])
    nz = np.clip(nz, 0, None)
    return z, nz / np.trapz(nz, z)


def camb_obs(cp, z, nz, bias, s, lmax, magnify):
    pars = camb.set_params(H0=cp["H0"], ombh2=cp["ombh2"], omch2=cp["omch2"],
                           ns=cp["ns"], As=cp["As"], tau=cp["tau"])
    win = SplinedSourceWindow(z=z, W=nz, source_type="counts", bias=bias, dlog10Ndm=s)
    pars.SourceWindows = [win]
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_lensing = bool(magnify)        # magnification term
    for t in _OFF:
        setattr(pars.SourceTerms, t, False)
    pars.SourceTerms.limber_windows = False
    pars.set_for_lmax(lmax + 50)
    d = camb.get_results(pars).get_cmb_unlensed_scalar_array_dict(CMB_unit="muK", raw_cl=True)
    ell = np.arange(lmax + 1)
    gT = d["TxW1"][: lmax + 1]
    gg = d["W1xW1"][: lmax + 1]
    kg = 0.5 * ell * (ell + 1) * d["PxW1"][: lmax + 1]     # phi g -> kappa g
    return ell, gT, gg, kg


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config)); cp = cosmo(cfg)
    bins = cfg["desi"]["bins"]
    slopes = dict(S_SLOPE)
    if a.slopes:
        slopes.update({k: float(v) for k, v in (x.split(":") for x in a.slopes.split(","))})

    # measured errors
    import glob
    bias = {}
    for bf in glob.glob(a.bias_glob):
        d = json.load(open(bf))
        T = d.get("tracer") or Path(bf).name.split("_")[0]
        for e in d.get("priors", []):
            bias[(T, int(e["ibin"]))] = (e["b"], e.get("sigma_b", np.nan))
    aerr = {}
    for f in glob.glob(a.aisw_glob):
        for r in csv.DictReader(open(f)):
            try:
                aerr[(r["tracer"], int(r["ibin"]))] = (float(r["A"]), float(r["sigmaA"]))
            except (ValueError, KeyError):
                pass

    band = lambda ell: (ell >= a.lmin) & (ell <= a.lmax)
    wfrac = lambda ell, dC, C, m: float(np.sum((2 * ell[m] + 1) * dC[m]) /
                                        np.sum((2 * ell[m] + 1) * C[m]))

    rows = []
    print(f"\nMagnification impact (s={slopes}, band ell=[{a.lmin},{a.lmax}])")
    print("  bias verdict: 'b:CHECK' if |db/b| > sigma_b/b (correct it); A verdict: 'A:CHECK' if "
          "|dA/A| > sigma_A/A")
    hdr = f"{'tracer':6s}{'bin':>4s}{'s':>6s} | {'dCgg/Cgg':>9s}{'dCkg/Ckg':>9s}{'dCgT/CgT':>9s}"\
          f" | {'db/b|kg':>8s}{'sig_b/b':>8s} | {'dA/A':>7s}{'sig_A/A':>8s}  verdict"
    print(hdr); print("-" * len(hdr))
    for T in bins:
        s = slopes.get(T, 0.4)
        for ib in range(1, len(bins[T]) + 1):
            p = Path(a.nz_dir) / f"{T}_z{ib}_nz.txt"
            if not p.exists():
                continue
            z, nz = read_nz(p)
            b = bias.get((T, ib), (2.0, np.nan))[0]
            ell, gT0, gg0, kg0 = camb_obs(cp, z, nz, b, s, a.lmax, magnify=False)
            _, gT1, gg1, kg1 = camb_obs(cp, z, nz, b, s, a.lmax, magnify=True)
            m = band(ell)
            f_gg = wfrac(ell, gg1 - gg0, gg0, m)
            f_kg = wfrac(ell, kg1 - kg0, kg0, m)
            f_gT = wfrac(ell, gT1 - gT0, gT0, m)
            db_b = f_kg                                   # kg pins b ~ linearly
            dA_A = f_gT - db_b
            sb_b = bias.get((T, ib), (np.nan, np.nan))[1] / b if (T, ib) in bias else np.nan
            A_, sA_ = aerr.get((T, ib), (np.nan, np.nan))
            sA_A = abs(sA_ / A_) if A_ and np.isfinite(A_) and A_ != 0 else np.nan
            vb = "b:CHECK" if (np.isfinite(sb_b) and abs(db_b) > sb_b) else "b:ok"
            va = "A:CHECK" if (np.isfinite(sA_A) and abs(dA_A) > sA_A) else "A:ok"
            verdict = f"{vb} {va}"
            print(f"{T:6s}{ib:>4d}{s:>6.2f} | {f_gg:>+9.1%}{f_kg:>+9.1%}{f_gT:>+9.1%}"
                  f" | {db_b:>+8.1%}{sb_b:>8.1%} | {dA_A:>+7.1%}{sA_A:>8.1%}  {verdict}")
            rows.append(dict(tracer=T, ibin=ib, s=s, dCgg=f_gg, dCkg=f_kg, dCgT=f_gT,
                             db_b=db_b, sigb_b=sb_b, dA_A=dA_A, sigA_A=sA_A, verdict=verdict))
    with open(a.out_csv, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n  wrote {a.out_csv}")
    nb = sum("b:CHECK" in r["verdict"] for r in rows)
    na = sum("A:CHECK" in r["verdict"] for r in rows)
    print(f"  bias:  {nb}/{len(rows)} bin(s) where magnification > sigma_b  -> correct b there")
    print(f"  A_ISW: {na}/{len(rows)} bin(s) where magnification > sigma_A  -> {'CHECK' if na else 'negligible for the headline'}\n")


if __name__ == "__main__":
    main()
