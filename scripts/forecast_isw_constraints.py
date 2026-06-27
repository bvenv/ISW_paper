#!/usr/bin/env python3
"""
forecast_isw_constraints.py

Forecast DESI×Planck ISW amplitude constraints for 4 tracer classes (BGS/LRG/ELG/QSO).

You can run in two modes:

  (A) DEMO mode (runnable immediately):
      --demo  (generates toy clTT + toy gT/gg templates and runs the forecast)

  (B) REAL mode (uses your theory templates):
      Provide templates:
        templates/isw/gT_BGS.npz ... gT_QSO.npz
        templates/isw/gg_BGS.npz ... gg_QSO.npz
        (optional) cross terms gg_BGS_LRG.npz, etc.
      Provide clTT:
        templates/planck_cltt.txt  (two columns: ell clTT), or .npz with ell/cl
      Provide f_sky:
        --fsky 0.30   (or pass --mask with healpy installed)

Theory format (NPZ):
  keys: 'ell' (int array), 'cl' (float array)

Forecast model:
  d_l,t =  C_l^{gT}(t) ~ A_t * C_l^{gT,fid}(t)
  Fisher: F_ij = sum_l t_i(l) [Cov_l^{-1}]_{ij} t_j(l)
  Cov_l (Gaussian):
    Cov_l(i,j) = 1/[(2l+1) f_sky] * [ C_l^{g_i g_j} * C_l^{TT} + C_l^{g_iT} C_l^{g_jT} ]
  Shot-noise can be added to gg auto terms if you provide nbar (sr^-1).

Outputs:
  prints sigma(A_t) and S/N (if A=1), writes CSV and NPZ with Cov(A), Corr(A).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np

# Optional
try:
    import yaml
except Exception:
    yaml = None

try:
    import healpy as hp
except Exception:
    hp = None


DEFAULT_TRACERS = ["BGS", "LRG", "ELG", "QSO"]


# ----------------------------
# I/O helpers
# ----------------------------
def save_npz(path: Path, ell: np.ndarray, cl: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ell=ell.astype(int), cl=cl.astype(float))


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["ell"].astype(int), d["cl"].astype(float)


def load_cltt(path: Path, ell: np.ndarray) -> np.ndarray:
    if path.suffix == ".npz":
        e0, c0 = load_npz(path)
    else:
        arr = np.loadtxt(path)
        e0, c0 = arr[:, 0].astype(int), arr[:, 1].astype(float)
    return np.interp(ell, e0, c0, left=0.0, right=0.0)


def load_templates(templates_dir: Path, tracers: list[str], ell: np.ndarray):
    # gT
    cl_gT: dict[str, np.ndarray] = {}
    for t in tracers:
        p = templates_dir / f"gT_{t}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing template: {p}")
        e0, c0 = load_npz(p)
        cl_gT[t] = np.interp(ell, e0, c0, left=0.0, right=0.0)

    # gg (auto required; cross optional)
    cl_gg: dict[tuple[str, str], np.ndarray] = {}
    for i, ti in enumerate(tracers):
        pa = templates_dir / f"gg_{ti}.npz"
        if not pa.exists():
            raise FileNotFoundError(f"Missing template: {pa}")
        e0, c0 = load_npz(pa)
        auto = np.interp(ell, e0, c0, left=0.0, right=0.0)
        cl_gg[(ti, ti)] = auto

        for tj in tracers[i + 1 :]:
            pc = templates_dir / f"gg_{ti}_{tj}.npz"
            if pc.exists():
                e1, c1 = load_npz(pc)
                cross = np.interp(ell, e1, c1, left=0.0, right=0.0)
            else:
                cross = np.zeros_like(ell, dtype=float)
            cl_gg[(ti, tj)] = cross
            cl_gg[(tj, ti)] = cross

    return cl_gT, cl_gg


def read_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml not installed. Either install pyyaml or avoid --config.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_nbar(path: Path) -> dict[str, float]:
    # supports YAML or JSON
    txt = path.read_text()
    try:
        d = json.loads(txt)
        return {str(k): float(v) for k, v in d.items()}
    except Exception:
        if yaml is None:
            raise RuntimeError("pyyaml not installed and nbar file is not JSON.")
        d = yaml.safe_load(txt)
        return {str(k): float(v) for k, v in d.items()}


def estimate_fsky_from_mask(mask_path: Path, nside: int) -> float:
    if hp is None:
        raise RuntimeError("healpy not available; pass --fsky instead of --mask.")
    w = hp.read_map(mask_path, dtype=np.float64, verbose=False)
    if hp.get_nside(w) != nside:
        w = hp.ud_grade(w, nside_out=nside, order_in="RING", order_out="RING", power=None)
    w = np.clip(w, 0.0, 1.0)
    # Effective f_sky for pseudo-Cl often uses <w^2>
    return float(np.mean(w**2))


# ----------------------------
# Forecast core
# ----------------------------
def build_cov_list(
    ell: np.ndarray,
    tracers: list[str],
    cltt: np.ndarray,
    cl_gT: dict[str, np.ndarray],
    cl_gg: dict[tuple[str, str], np.ndarray],
    fsky: float,
    nbar_sr: dict[str, float] | None = None,
) -> list[np.ndarray]:
    """Per-ell covariance matrices for the vector d_l = [C_l^{gT}(tracer_1),...,C_l^{gT}(tracer_N)]."""
    n = len(tracers)
    cov_list: list[np.ndarray] = []

    shot = {}
    if nbar_sr:
        for t, nb in nbar_sr.items():
            if nb > 0:
                shot[t] = 1.0 / nb

    for k, l in enumerate(ell):
        if l < 2:
            cov_list.append(np.full((n, n), np.inf))
            continue

        pref = 1.0 / ((2.0 * l + 1.0) * fsky)
        C = np.zeros((n, n), dtype=float)

        for i, ti in enumerate(tracers):
            cgiT = cl_gT[ti][k]
            for j, tj in enumerate(tracers):
                cgjT = cl_gT[tj][k]
                cgg = cl_gg[(ti, tj)][k]
                if ti == tj and ti in shot:
                    cgg = cgg + shot[ti]
                C[i, j] = pref * (cgg * cltt[k] + cgiT * cgjT)

        cov_list.append(C)

    return cov_list


def fisher_A_per_tracer(
    ell: np.ndarray,
    tracers: list[str],
    cl_gT: dict[str, np.ndarray],
    cov_list: list[np.ndarray],
    lmin: int,
    lmax: int,
) -> np.ndarray:
    """Fisher matrix for A_t parameters (one amplitude per tracer)."""
    n = len(tracers)
    F = np.zeros((n, n), dtype=float)

    for k, l in enumerate(ell):
        if l < lmin or l > lmax:
            continue
        C = cov_list[k]
        if not np.all(np.isfinite(C)):
            continue

        # Stabilise inversion a bit
        try:
            Ci = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Ci = np.linalg.pinv(C, rcond=1e-12)

        tvec = np.array([cl_gT[t][k] for t in tracers], dtype=float)

        # F_ij += t_i * Ci_ij * t_j
        F += (tvec[:, None] * tvec[None, :]) * Ci

    return F


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (d[:, None] * d[None, :])
    corr[np.isnan(corr)] = 0.0
    return corr


# ----------------------------
# Demo generator (toy templates)
# ----------------------------
def make_demo_files(out_dir: Path, tracers: list[str], lmax: int) -> tuple[Path, Path]:
    """
    Creates:
      out_dir/cltt_demo.txt
      out_dir/gT_<tracer>.npz
      out_dir/gg_<tracer>.npz
      out_dir/gg_<tr1>_<tr2>.npz
    Returns (templates_dir, cltt_path).
    """
    templates_dir = out_dir / "templates_demo"
    templates_dir.mkdir(parents=True, exist_ok=True)

    ell = np.arange(0, lmax + 1, dtype=int)

    # Toy clTT (not physical; good enough to test numerics)
    # ~ l^{-2} on large scales
    cltt = np.zeros_like(ell, dtype=float)
    good = ell >= 2
    cltt[good] = 1e-10 * (ell[good] / 30.0) ** (-2.0)
    cltt_path = out_dir / "cltt_demo.txt"
    np.savetxt(cltt_path, np.c_[ell, cltt], fmt=["%d", "%.10e"])

    # Toy templates: gT peaked at low ell; gg ~ l^{-1}
    amps = {"BGS": 1.0, "LRG": 0.9, "ELG": 0.7, "QSO": 0.5}
    gg_amp = {"BGS": 2.0, "LRG": 1.6, "ELG": 1.2, "QSO": 1.0}
    rho = 0.25  # cross-correlation coefficient for gg cross terms (toy)

    def lowell_bump(l, l0=35.0, width=45.0):
        x = l / width
        return np.exp(-(x**2)) * (l0 / np.maximum(l, 2)) ** 0.7

    for t in tracers:
        gT = np.zeros_like(ell, dtype=float)
        gT[good] = 5e-12 * amps.get(t, 0.8) * lowell_bump(ell[good])
        save_npz(templates_dir / f"gT_{t}.npz", ell, gT)

        gg = np.zeros_like(ell, dtype=float)
        gg[good] = 2e-7 * gg_amp.get(t, 1.0) * (ell[good] / 30.0) ** (-1.1)
        save_npz(templates_dir / f"gg_{t}.npz", ell, gg)

    # Cross gg terms (toy)
    for i, ti in enumerate(tracers):
        ei, ggi = load_npz(templates_dir / f"gg_{ti}.npz")
        for tj in tracers[i + 1 :]:
            ej, ggj = load_npz(templates_dir / f"gg_{tj}.npz")
            cross = rho * np.sqrt(ggi * ggj)
            save_npz(templates_dir / f"gg_{ti}_{tj}.npz", ell, cross)

    return templates_dir, cltt_path

import numpy as np
import matplotlib.pyplot as plt

def _gaussian_pdf(x, mu, sig):
    sig = float(sig)
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)

def plot_A_posteriors(tracer_names, sigmas, mu=1.0, nsig=4.0,
                      ridge=False, outpath="isw_A_posteriors.png",
                      add_combined=True):
    """
    Make a Dong+2022-style PDF plot: Gaussian posteriors, peak-normalized,
    for each tracer/bin. Optionally stacked (ridgeline) to avoid clutter.
    """
    tracer_names = list(tracer_names)
    sigmas = np.asarray(sigmas, dtype=float)

    if len(tracer_names) != len(sigmas):
        raise ValueError("tracer_names and sigmas must have the same length")

    sig_max = np.max(sigmas)
    x_lo = mu - nsig * sig_max
    x_hi = mu + nsig * sig_max
    x = np.linspace(x_lo, x_hi, 1200)

    fig = plt.figure(figsize=(8.0, 3.0))  # slide-friendly aspect
    ax = plt.gca()

    if ridge:
        # Ridgeline: each PDF is vertically offset; still "one axis" (one panel).
        dy = 1.15
        for i, (name, s) in enumerate(zip(tracer_names, sigmas)):
            pdf = _gaussian_pdf(x, mu, s)
            pdf /= np.max(pdf)  # peak-normalize (Dong-style)
            y0 = i * dy
            ax.plot(x, pdf + y0, lw=2)
            ax.text(x_hi, y0 + 0.5, f"{name}  ($\\sigma={s:.3g}$)",
                    va="center", ha="right", fontsize=10)
        ax.set_yticks([])
        ax.set_ylim(-0.1, (len(tracer_names) - 1) * dy + 1.2)
        ax.set_ylabel("")  # keep clean for slides
    else:
        # Overlaid curves: simplest.
        for name, s in zip(tracer_names, sigmas):
            pdf = _gaussian_pdf(x, mu, s)
            pdf /= np.max(pdf)
            ax.plot(x, pdf, lw=2, label=f"{name} ($\\sigma={s:.3g}$)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("peak-normalized PDF")
        ax.legend(frameon=False, fontsize=9, ncol=2)

    # Reference line at fiducial A=1
    ax.axvline(mu, ls="--", lw=1)
    ax.set_xlabel(r"$A\ ( \mathrm{ISW\ amplitude})$")

    # Optional “combined” constraint (inverse-variance combo; assumes independence)
    if add_combined and len(sigmas) > 1:
        ivar = np.sum(1.0 / sigmas**2)
        sig_c = np.sqrt(1.0 / ivar)
        pdf_c = _gaussian_pdf(x, mu, sig_c)
        pdf_c /= np.max(pdf_c)
        if ridge:
            # put at bottom, lightly separated
            y0 = -0.9
            ax.plot(x, pdf_c + y0, lw=2, ls=":")
            ax.text(x_hi, y0 + 0.5, f"combined  ($\\sigma={sig_c:.3g}$)",
                    va="center", ha="right", fontsize=10)
            ax.set_ylim(y0 - 0.1, ax.get_ylim()[1])
        else:
            ax.plot(x, pdf_c, lw=2, ls=":", label=f"combined ($\\sigma={sig_c:.3g}$)")
            ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def _ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_theory_cls(outdir: Path, ell: np.ndarray, cltt: np.ndarray,
                    cl_gT: dict[str, np.ndarray],
                    cl_gg: dict[tuple[str, str], np.ndarray],
                    tracers: list[str]) -> None:
    import matplotlib.pyplot as plt

    outdir = _ensure_outdir(outdir)

    # TT
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = plt.gca()
    m = ell >= 2
    ax.loglog(ell[m], cltt[m], lw=2)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{TT}$  [$\mu{\rm K}^2$]")
    fig.tight_layout()
    fig.savefig(outdir / "clTT.png", dpi=200)
    plt.close(fig)

    # gT (note: can be negative; use abs + sign markers)
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = plt.gca()
    for t in tracers:
        y = cl_gT[t][m]
        ax.loglog(ell[m], np.abs(y) + 1e-40, lw=2, label=f"{t}  (|gT|)")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$|C_\ell^{gT}|$  [$\mu{\rm K}$]")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "cl_gT_abs.png", dpi=200)
    plt.close(fig)

    # gg autos
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = plt.gca()
    for t in tracers:
        y = cl_gg[(t, t)][m]
        ax.loglog(ell[m], y + 1e-40, lw=2, label=f"{t}×{t}")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{gg}$")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "cl_gg_auto.png", dpi=200)
    plt.close(fig)

def plot_cov_corr_at_ell(outdir: Path, ell: np.ndarray, cov_list: list[np.ndarray],
                         tracers: list[str], ells_to_plot=(10, 30, 80, 150)) -> None:
    import matplotlib.pyplot as plt

    outdir = _ensure_outdir(outdir)
    name_to_i = {t: i for i, t in enumerate(tracers)}

    for L in ells_to_plot:
        if L < ell.min() or L > ell.max():
            continue
        k = int(L)
        C = cov_list[k]
        d = np.sqrt(np.diag(C))
        with np.errstate(divide="ignore", invalid="ignore"):
            R = C / (d[:, None] * d[None, :])
        R[np.isnan(R)] = 0.0

        fig = plt.figure(figsize=(4.6, 4.2))
        ax = plt.gca()
        im = ax.imshow(R, vmin=-1, vmax=1)
        ax.set_xticks(range(len(tracers)))
        ax.set_yticks(range(len(tracers)))
        ax.set_xticklabels(tracers, rotation=45, ha="right")
        ax.set_yticklabels(tracers)
        ax.set_title(f"Corr[ C_ell^{{gT}} ] at ell={L}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"corr_cov_ell{L}.png", dpi=200)
        plt.close(fig)

def plot_cumulative_snr(outdir: Path, ell: np.ndarray, tracers: list[str],
                        cl_gT: dict[str, np.ndarray], cov_list: list[np.ndarray],
                        lmin: int, lmax: int) -> None:
    import matplotlib.pyplot as plt

    outdir = _ensure_outdir(outdir)
    m = (ell >= lmin) & (ell <= lmax)
    ells = ell[m]

    # Fisher contribution per-ell: F_l = t t^T C^{-1}
    # We track diagonals as “per-tracer SNR^2 contributions” for A_t.
    snr2 = {t: np.zeros_like(ells, dtype=float) for t in tracers}

    for ii, L in enumerate(ells):
        C = cov_list[int(L)]
        try:
            Ci = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Ci = np.linalg.pinv(C, rcond=1e-12)

        tvec = np.array([cl_gT[t][int(L)] for t in tracers], dtype=float)
        Fl = (tvec[:, None] * tvec[None, :]) * Ci  # (n,n)

        for j, t in enumerate(tracers):
            snr2[t][ii] = Fl[j, j]

    fig = plt.figure(figsize=(7.2, 3.8))
    ax = plt.gca()
    for t in tracers:
        ax.plot(ells, np.cumsum(snr2[t]), lw=2, label=t)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"cumulative SNR$^2$ (diag Fisher)")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "cumulative_snr2.png", dpi=200)
    plt.close(fig)


import numpy as np

def snr2_eq5_isw_upper(ell, cltt_isw, cltt_tot, fsky=1.0, ell_min=2, ell_max=None):
    """
    Eq. (5): (S/N)^2 <= sum_l (2l+1) * C_l^{TT,ISW} / C_l^{TT,tot}
    Inputs must be consistent units (e.g. uK^2 for both cltt arrays).
    """
    ell = np.asarray(ell)
    m = ell >= ell_min
    if ell_max is not None:
        m &= (ell <= ell_max)

    num = (2.0 * ell[m] + 1.0) * fsky * cltt_isw[m]
    den = cltt_tot[m]
    good = (den > 0) & np.isfinite(num) & np.isfinite(den)
    return float(np.sum(num[good] / den[good]))


def snr2_eq6_tg(ell, cl_tg, cltt_tot, clgg, fsky=0.3, ell_min=2, ell_max=None, nbar_sr=None):
    """
    Eq. (6): (S/N)^2 ~= f_sky * sum_l (2l+1) * (C_l^{Tg})^2 /
              [ (C_l^{Tg})^2 + C_l^{TT,tot} (C_l^{gg} + 1/n_s) ]

    Units:
      - cltt_tot: temperature^2  (e.g. uK^2)
      - cl_tg:    temperature    (e.g. uK)
      - clgg:     dimensionless
      - nbar_sr:  number per steradian (so 1/nbar_sr is dimensionless)
    """
    ell = np.asarray(ell)
    m = ell >= ell_min
    if ell_max is not None:
        m &= (ell <= ell_max)

    nl = 0.0 if (nbar_sr is None) else (1.0 / float(nbar_sr))

    ctg2 = cl_tg[m] ** 2
    denom = ctg2 + cltt_tot[m] * (clgg[m] + nl)
    # print(clgg)
    good = (denom > 0) & np.isfinite(ctg2) & np.isfinite(denom)
    w = (2.0 * ell[m] + 1.0) * fsky
    return float(np.sum(w[good] * ctg2[good] / denom[good]))


def pretty_snr(label, snr2):
    snr = np.sqrt(max(snr2, 0.0))
    sigA = np.inf if snr == 0 else (1.0 / snr)  # if parameter is an overall amplitude A with fid A=1
    return f"{label:>10s}: S/N = {snr:6.3f}   (sigma_A ~ {sigA:7.4f})"



# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracers", nargs="+", default=DEFAULT_TRACERS, help="Tracer list (default: BGS LRG ELG QSO)")
    ap.add_argument("--ell-min", type=int, default=2)
    ap.add_argument("--ell-max", type=int, default=150)
    ap.add_argument("--fsky", type=float, default=None, help="Sky fraction. If omitted, use --mask to estimate.")
    ap.add_argument("--mask", type=str, default=None, help="Joint mask FITS (healpy required) to estimate f_sky.")
    ap.add_argument("--nside", type=int, default=1024)

    # Real mode inputs
    ap.add_argument("--templates-dir", type=str, default=None, help="Directory containing gT_*.npz and gg_*.npz")
    ap.add_argument("--cltt", type=str, default=None, help="clTT file (.txt two cols or .npz with ell/cl)")

    # Optional config / nbar
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config (not required)")
    ap.add_argument("--nbar", type=str, default=None, help="Optional YAML/JSON mapping tracer->nbar_sr")

    # Outputs
    ap.add_argument("--out-csv", type=str, default="results/isw_forecast_A_4tracers.csv")
    ap.add_argument("--out-npz", type=str, default="results/isw_forecast_A_4tracers.npz")

    # Demo mode
    ap.add_argument("--demo", action="store_true", help="Generate toy templates + clTT and run end-to-end.")
    ap.add_argument("--demo-dir", type=str, default="demo_forecast_outputs")
    ap.add_argument("--plot-posteriors", action="store_true",
                help="Plot forecast posteriors for A(z) in each bin/tracer.")
    ap.add_argument("--ridge", action="store_true",
                    help="Use ridgeline (stacked) PDFs instead of overlaid curves.")
    ap.add_argument("--fig", default="isw_A_posteriors.png",
                    help="Output figure filename (saved in CWD or --outdir if you have one).")
    ap.add_argument("--diag-plots", action="store_true",
                    help="Write diagnostic plots (cls, cov-corr, cumulative snr) to --diag-outdir.")
    ap.add_argument("--diag-outdir", default="results/diag_isw",
                    help="Output directory for --diag-plots.")



    args = ap.parse_args()

    tracers = list(args.tracers)
    ell = np.arange(0, args.ell_max + 1, dtype=int)

    # Determine f_sky
    if args.fsky is not None:
        fsky = float(args.fsky)
    else:
        if args.mask is None:
            raise SystemExit("Need --fsky or --mask (healpy) to proceed.")
        fsky = estimate_fsky_from_mask(Path(args.mask), args.nside)

    # Optional config read (not required)
    if args.config:
        _ = read_config(Path(args.config))

    # Optional shot noise
    nbar_sr = read_nbar(Path(args.nbar)) if args.nbar else None

    # Demo vs real
    if args.demo:
        demo_dir = Path(args.demo_dir)
        templates_dir, cltt_path = make_demo_files(demo_dir, tracers, args.ell_max)
        print(f"[demo] wrote templates to: {templates_dir}")
        print(f"[demo] wrote clTT to:      {cltt_path}")
    else:
        if args.templates_dir is None or args.cltt is None:
            raise SystemExit("Real mode requires --templates-dir and --cltt (or use --demo).")
        templates_dir = Path(args.templates_dir)
        cltt_path = Path(args.cltt)

    # Load theory pieces
    cltt = load_cltt(cltt_path, ell)
    cl_gT, cl_gg = load_templates(templates_dir, tracers, ell)

    # --- Eq. (6) S/N per tracer (single-tracer approximation) ---
    print("\n[Sanity] Eq. (6) single-tracer S/N estimates:")
    for t in tracers:
        # print(nbar_sr)
        snr2 = snr2_eq6_tg(
            ell,
            cl_tg=cl_gT[t],
            cltt_tot=cltt,
            clgg=cl_gg[(t, t)],
            fsky=fsky,
            ell_min=args.ell_min,
            ell_max=args.ell_max,
            nbar_sr=(None if nbar_sr is None else nbar_sr.get(t, None)),
        )
        # print(cl_gT[t],cltt)
        print(pretty_snr(t, snr2))

        r = cl_gT[t] / np.sqrt(np.maximum(cltt * cl_gg[(t, t)], 1e-300))
        # print(f"  {t}  r_ell min/median/max = {np.nanmin(r):.3g} / {np.nanmedian(r):.3g} / {np.nanmax(r):.3g}")



    # Build per-ell covariance and Fisher
    cov_list = build_cov_list(ell, tracers, cltt, cl_gT, cl_gg, fsky, nbar_sr=nbar_sr)
    F = fisher_A_per_tracer(ell, tracers, cl_gT, cov_list, args.ell_min, args.ell_max)

    # Invert Fisher
    try:
        covA = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        covA = np.linalg.pinv(F, rcond=1e-12)

    sigA = np.sqrt(np.diag(covA))
    corrA = corr_from_cov(covA)

    print(f"f_sky ≈ {fsky:.4f}  |  ell range = [{args.ell_min}, {args.ell_max}]")
    for t, s in zip(tracers, sigA):
        sn = (1.0 / s) if s > 0 else np.nan
        print(f"  sigma(A_{t}) = {s:.4g}   (S/N ~ {sn:.3g} if A=1)")

    # Save outputs
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("tracer,sigmaA,SN_if_A1\n")
        for t, s in zip(tracers, sigA):
            sn = (1.0 / s) if s > 0 else np.nan
            f.write(f"{t},{s},{sn}\n")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, tracers=np.array(tracers), covA=covA, corrA=corrA, fsky=fsky,
             ell_min=args.ell_min, ell_max=args.ell_max)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_npz}")

    if args.plot_posteriors:
        # Optional: prettier labels for the slide
        pretty = {
            "BGS": "BGS",
            "LRG": "LRG",
            "ELG": "ELG",
            "QSO": "QSO",
        }
        tracer_names = [pretty.get(t, t) for t in tracers]
        sigmas = sigA  # already ordered to match `tracers`

        # Put the figure next to your outputs by default
        fig_path = Path(args.fig)
        if fig_path.parent == Path("."):
            fig_path = Path("results") / fig_path
        fig_path.parent.mkdir(parents=True, exist_ok=True)

        outfig = plot_A_posteriors(
            tracer_names=tracer_names,
            sigmas=sigmas,
            mu=1.0,
            ridge=args.ridge,
            outpath=str(fig_path),
            add_combined=True,
        )
        print(f"[saved] {outfig}")

    if args.diag_plots:
        ddir = Path(args.diag_outdir)
        plot_theory_cls(ddir, ell, cltt, cl_gT, cl_gg, tracers)
        plot_cov_corr_at_ell(ddir, ell, cov_list, tracers)
        plot_cumulative_snr(ddir, ell, tracers, cl_gT, cov_list, args.ell_min, args.ell_max)
        print(f"[diag] wrote plots to: {ddir}")




if __name__ == "__main__":
    main()
