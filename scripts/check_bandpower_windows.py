#!/usr/bin/env python3
"""
Validate the (2l+1) boxcar theory binning (`bandaverage`) against the EXACT NaMaster bandpower
window W_bl (from the cached workspace) for the gT/ISW templates, and propagate the difference to
the joint A_ISW. Confirms whether the boxcar approximation is good enough or the exact window is
needed. Proper binning: C_b = decouple_cell(couple_cell(C_l^theory)).
"""
import argparse, glob
from pathlib import Path
import numpy as np
import yaml
import pymaster as nmt

from magnification_impact import cosmo, read_nz
from build_isw_templates_clank import camb_window_cls, load_bias_priors, BIAS_FALLBACK
from fit_isw_amplitudes import bandaverage


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--nz-dir", default="templates/isw/nz")
    p.add_argument("--bias-priors", default="results/desi_bias_kappa.json")
    p.add_argument("--lmax", type=int, default=150)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config)); cp = cosmo(cfg)
    spectra = Path(cfg["paths"]["spectra"]) / "desi"
    biasmap = load_bias_priors(a.bias_priors)
    sc = np.load("results/sim_cov_all.npz", allow_pickle=True)
    Sigma, nsims = sc["Sigma"], int(sc["nsims"])
    tr_lay = [str(x) for x in sc["tracers"]]; ib_lay = [int(x) for x in sc["ibins"]]
    covs = {T: np.load(f"results/sim_cov_{T}.npz", allow_pickle=True) for T in set(tr_lay)
            if Path(f"results/sim_cov_{T}.npz").exists()}

    wsp_cache = {}
    def get_wsp(path):
        if path not in wsp_cache:
            w = nmt.NmtWorkspace(); w.read_from(path); wsp_cache[path] = w
        return wsp_cache[path]

    print("\nBandpower-window check: boxcar `bandaverage` vs exact NaMaster window (gT templates)")
    print(f"  {'tracer':6s}{'bin':>4s}  {'max|Δ|/box (l<60)':>18s}  {'A_box':>7s}{'A_exact':>8s}{'Δrel':>7s}")
    A_box = np.full(len(tr_lay), np.nan); A_exa = np.full(len(tr_lay), np.nan)
    maxdiffs = []
    for j, (T, ib) in enumerate(zip(tr_lay, ib_lay)):
        sp = spectra / f"gT_{T}_z{ib}_lmax{a.lmax}.npz"
        if not sp.exists():
            continue
        spec = np.load(sp, allow_pickle=True)
        d = spec["cl"]; lo, hi = spec["ell_edges_lo"], spec["ell_edges_hi"]
        wsp = get_wsp(str(spec["wsp_path"]))
        nl = wsp.wsp.lmax + 1 if hasattr(wsp, "wsp") else 3 * 512
        z, nz = read_nz(Path(a.nz_dir) / f"{T}_z{ib}_nz.txt")
        b0, al = BIAS_FALLBACK.get(T, (2.0, 0.6))
        b = biasmap.get((T, ib), b0 * (1 + 0.5 * z.mean()) ** al)
        ell, gT, _ = camb_window_cls(cp, z, nz, b, a.lmax)
        # full-length theory (zero-pad beyond lmax; ISW gT ~ 0 there)
        cl_full = np.zeros(nl); n = min(len(gT), nl); cl_full[:n] = gT[:n]
        t_box = bandaverage(ell, gT, lo, hi)
        t_exa = wsp.decouple_cell(wsp.couple_cell(np.array([cl_full])))[0][:len(d)]
        m = 0.5 * (lo + hi) < 60
        maxd = float(np.max(np.abs(t_exa - t_box)[m]) / np.max(np.abs(t_box[m])))
        maxdiffs.append(maxd)
        nb_ = int(covs[T]["nband"]); blk = covs[T]["cov"][(ib-1)*nb_:ib*nb_, (ib-1)*nb_:ib*nb_]
        Cinv = (nsims - len(d) - 2) / (nsims - 1) * np.linalg.inv(blk)
        A_box[j] = float(t_box @ Cinv @ d) / float(t_box @ Cinv @ t_box)
        A_exa[j] = float(t_exa @ Cinv @ d) / float(t_exa @ Cinv @ t_exa)
        print(f"  {T:6s}{ib:>4d}  {maxd:>17.1%}  {A_box[j]:>7.2f}{A_exa[j]:>8.2f}"
              f"{(A_exa[j]-A_box[j])/A_box[j]:>+7.1%}")

    def joint(A):
        mm = np.isfinite(A)
        S = Sigma[np.ix_(np.where(mm)[0], np.where(mm)[0])]
        Sinv = (nsims - mm.sum() - 2) / (nsims - 1) * np.linalg.inv(S)
        one = np.ones(mm.sum()); den = float(one @ Sinv @ one)
        return float((one @ Sinv @ A[mm]) / den), float(1 / np.sqrt(den))

    Ab, sb = joint(A_box); Ae, se = joint(A_exa)
    print(f"\n  median |Δ|/boxcar over the bandpowers (ℓ<60): {np.median(maxdiffs):.1%}")
    print(f"  joint A_ISW:  boxcar = {Ab:.3f} ± {sb:.3f};  exact window = {Ae:.3f}"
          f"  -> shift {Ae-Ab:+.3f} ({(Ae-Ab)/sb:+.2f} σ_A)")
    print(f"  => boxcar binning {'adequate' if abs(Ae-Ab) < 0.5*sb else 'CHECK — use exact window'}\n")


if __name__ == "__main__":
    main()
