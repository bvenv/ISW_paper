#!/usr/bin/env python3
"""
Level-2 magnification correction to the κg-pinned bias b(z).

The magnification template pieces are bias-INDEPENDENT, so we compute them once per tracer x bin
from CAMB source windows (three runs: density-only, magnification-only, both) and correct the
saved unit-bias amplitudes exactly:

    kg:  A_kg = b + r_kg                  with r_kg  = <T_kmu>/<T_kd>
    gg:  A_gg = b^2 + 2 b r_dmu + r_mumu  with r_dmu = <T_dmu>/<T_dd>, r_mumu = <T_mumu>/<T_dd>
    ->   b_kg_corr = A_kg - r_kg ;   b_gg_corr = -r_dmu + sqrt(r_dmu^2 + A_gg - r_mumu)

The combined b is reformed with the SAME gg/kg weight the original fit used (recovered from the
stored b, b_gg, b_kg); statistical sigma_b is unchanged (magnification is a central-value shift).
Reads results/bias/{T}_bias_kappa.json, writes results/bias/{T}_bias_magcorr.json (originals kept).
"""
import argparse, csv, json, glob
from pathlib import Path
import numpy as np
import yaml
import camb
from camb.sources import SplinedSourceWindow

from magnification_impact import S_SLOPE, cosmo, read_nz, _OFF


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--bias-glob", default="results/bias/*_bias_kappa.json")
    p.add_argument("--lmin", type=int, default=2)
    p.add_argument("--lmax", type=int, default=150)
    p.add_argument("--slopes", default=None)
    p.add_argument("--out-csv", default="results/magnification_bias_correction.csv")
    p.add_argument("--out-plot", default="results/plots/bias_magnification_corrected.png")
    return p.parse_args()


def camb_pieces(cp, z, nz, s, lmax, density, lensing):
    pars = camb.set_params(H0=cp["H0"], ombh2=cp["ombh2"], omch2=cp["omch2"],
                           ns=cp["ns"], As=cp["As"], tau=cp["tau"])
    pars.SourceWindows = [SplinedSourceWindow(z=z, W=nz, source_type="counts", bias=1.0, dlog10Ndm=s)]
    pars.SourceTerms.counts_density = bool(density)
    pars.SourceTerms.counts_lensing = bool(lensing)
    for t in _OFF:
        setattr(pars.SourceTerms, t, False)
    pars.SourceTerms.limber_windows = False
    pars.set_for_lmax(lmax + 50)
    d = camb.get_results(pars).get_cmb_unlensed_scalar_array_dict(CMB_unit="muK", raw_cl=True)
    ell = np.arange(lmax + 1)
    gg = d["W1xW1"][: lmax + 1]
    kg = 0.5 * ell * (ell + 1) * d["PxW1"][: lmax + 1]
    return ell, gg, kg


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config)); cp = cosmo(cfg)
    bins = cfg["desi"]["bins"]
    slopes = dict(S_SLOPE)
    if a.slopes:
        slopes.update({k: float(v) for k, v in (x.split(":") for x in a.slopes.split(","))})

    wband = None  # (2l+1) weight over the band, set per call
    def bavg(ell, C):
        m = (ell >= a.lmin) & (ell <= a.lmax)
        w = 2 * ell[m] + 1
        return float(np.sum(w * C[m]) / np.sum(w))

    rows = []
    print(f"\nLevel-2 magnification correction to b(z)  (band ell=[{a.lmin},{a.lmax}])")
    print(f"  {'tracer':6s}{'bin':>4s}{'s':>6s} | {'b_old':>7s}{'b_new':>7s}{'Δb':>7s}{'Δb/σ_b':>8s}"
          f" | {'b_gg':>6s}->{'':>5s}{'b_kg':>7s}->")
    for bf in sorted(glob.glob(a.bias_glob)):
        d = json.load(open(bf))
        T = d.get("tracer") or Path(bf).name.split("_")[0]
        s = slopes.get(T, 0.4)
        for e in d["priors"]:
            ib = int(e["ibin"])
            p = Path(a.nz_dir) / f"{T}_z{ib}_nz.txt"
            if not p.exists():
                continue
            z, nz = read_nz(p)
            ell, gg_d, kg_d = camb_pieces(cp, z, nz, s, a.lmax, density=True, lensing=False)
            _, gg_m, kg_m = camb_pieces(cp, z, nz, s, a.lmax, density=False, lensing=True)
            _, gg_b, _ = camb_pieces(cp, z, nz, s, a.lmax, density=True, lensing=True)
            Tdd, Tkd = bavg(ell, gg_d), bavg(ell, kg_d)
            Tmm, Tkm = bavg(ell, gg_m), bavg(ell, kg_m)
            Tdm = 0.5 * (bavg(ell, gg_b) - Tdd - Tmm)
            r_kg, r_dm, r_mm = Tkm / Tkd, Tdm / Tdd, Tmm / Tdd

            A_gg, A_kg = float(e["A_gg"]), float(e["A_kg"])
            b_old, sb = float(e["b"]), float(e.get("sigma_b", np.nan))
            bgg_o, bkg_o = float(e.get("b_gg", np.sqrt(A_gg))), float(e.get("b_kg", A_kg))
            # corrected single-probe biases
            bkg_c = A_kg - r_kg
            disc = r_dm ** 2 + A_gg - r_mm
            bgg_c = -r_dm + np.sqrt(disc) if disc > 0 else np.nan
            # preserve the original gg/kg combination weight
            w = (b_old - bkg_o) / (bgg_o - bkg_o) if abs(bgg_o - bkg_o) > 1e-6 else 0.5
            w = min(max(w, 0.0), 1.0)
            # fall back to the single well-defined probe if the other branch is undefined
            if not np.isfinite(bgg_c):
                w = 0.0
            elif not np.isfinite(bkg_c):
                w = 1.0
            b_new = bkg_c if w == 0.0 else bgg_c if w == 1.0 else w * bgg_c + (1 - w) * bkg_c
            db = b_new - b_old
            print(f"  {T:6s}{ib:>4d}{s:>6.2f} | {b_old:>7.3f}{b_new:>7.3f}{db:>+7.3f}"
                  f"{(db/sb if np.isfinite(sb) else np.nan):>+8.2f} | {bgg_o:>6.2f}->{bgg_c:>5.2f}"
                  f"{bkg_o:>7.2f}->{bkg_c:>.2f}")
            rows.append(dict(tracer=T, ibin=ib, s=s, b_old=b_old, b_new=b_new, db=db,
                             db_over_sigma=(db / sb if np.isfinite(sb) else np.nan),
                             sigma_b=sb, b_gg_old=bgg_o, b_gg_new=bgg_c,
                             b_kg_old=bkg_o, b_kg_new=bkg_c, w_gg=w,
                             r_kg=r_kg, r_dmu=r_dm, r_mumu=r_mm))

    with open(a.out_csv, "w", newline="") as fo:
        wcsv = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    # write corrected per-tracer bias JSONs (originals untouched)
    for bf in sorted(glob.glob(a.bias_glob)):
        d = json.load(open(bf)); T = d.get("tracer") or Path(bf).name.split("_")[0]
        for e in d["priors"]:
            rr = next((r for r in rows if r["tracer"] == T and r["ibin"] == int(e["ibin"])), None)
            if rr:
                e["b_uncorrected"] = e["b"]; e["b"] = round(rr["b_new"], 5)
                e["magnification_s"] = rr["s"]
        json.dump(d, open(bf.replace("_bias_kappa.json", "_bias_magcorr.json"), "w"), indent=2)

    # before/after plot
    import matplotlib; matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    COL = {"BGS": "#d62728", "LRG": "#1f77b4", "ELG": "#2ca02c", "QSO": "#9467bd"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for T in {r["tracer"] for r in rows}:
        sub = [r for r in rows if r["tracer"] == T]
        zc = [0.5 * sum(bins[T][r["ibin"] - 1]) for r in sub]
        ax.errorbar(zc, [r["b_old"] for r in sub], yerr=[r["sigma_b"] for r in sub], fmt="o:",
                    color=COL.get(T, "k"), alpha=.45, mfc="none")
        ax.errorbar(zc, [r["b_new"] for r in sub], yerr=[r["sigma_b"] for r in sub], fmt="o-",
                    color=COL.get(T, "k"), label=T)
    ax.plot([], [], "ko:", mfc="none", alpha=.5, label="uncorrected")
    ax.plot([], [], "ks-", label="magnification-corrected")
    ax.set_xlabel(r"$\bar z_{\rm bin}$"); ax.set_ylabel("linear bias $b$")
    ax.set_title("DESI bias: magnification-corrected vs uncorrected")
    ax.grid(ls=":", alpha=.4); ax.legend(fontsize=8, ncol=2, frameon=False)
    Path(a.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out_plot, dpi=150); plt.close(fig)
    print(f"\n  wrote {a.out_csv}, {a.out_plot}, results/bias/*_bias_magcorr.json")
    big = [r for r in rows if abs(r["db_over_sigma"]) > 1]
    print(f"  {len(big)}/{len(rows)} bins shift b by >1σ_b: "
          f"{', '.join(f'{r['tracer']}z{r['ibin']}({r['db_over_sigma']:+.1f}σ)' for r in big)}\n")


if __name__ == "__main__":
    main()
