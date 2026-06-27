#!/usr/bin/env python3
"""
Generate the DESI × Planck Cross-Correlation Program design document (PDF).

Self-contained (matplotlib only): multi-page letter-size PDF with project structure,
a pipeline flowchart, theory + mathematics, the script map, the clank-v4 fixes, current
status/results, the forecast/strategy, and the new (κg + ISW) plan.

  python scripts/make_project_doc.py --out DESI_xcorr_Program.pdf
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

NAVY = "#1f3b6f"
GREY = "#666666"


# ---------------------------------------------------------------- page renderer
def render_page(pdf, title, elements, page_no, total, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.5, 0.965, title, ha="center", va="center", fontsize=16.5,
            fontweight="bold", color="black", wrap=True)
    if subtitle:
        ax.text(0.5, 0.935, subtitle, ha="center", va="center", fontsize=10,
                color=GREY, style="italic")
    ax.plot([0.08, 0.92], [0.915, 0.915], color="#333", lw=1.0)

    y = 0.89
    for el in elements:
        kind = el[0]
        if kind == "h":              # heading
            y -= 0.006
            ax.text(0.09, y, el[1], fontsize=12, fontweight="bold", va="top", color=NAVY)
            y -= 0.027
        elif kind == "p":            # paragraph
            w = textwrap.fill(el[1], width=96)
            ax.text(0.09, y, w, fontsize=9.6, va="top", color="black")
            y -= 0.0150 * (w.count("\n") + 1) + 0.006
        elif kind == "eq":           # display equation (mathtext)
            ax.text(0.5, y, f"${el[1]}$", fontsize=12.5, ha="center", va="top")
            y -= 0.040
        elif kind == "b":            # bullet list
            for it in el[1]:
                w = textwrap.fill(it, width=90)
                ax.text(0.105, y, "•", fontsize=9.6, va="top", color=NAVY)
                ax.text(0.125, y, w, fontsize=9.6, va="top", color="black")
                y -= 0.0150 * (w.count("\n") + 1) + 0.004
            y -= 0.004
        elif kind == "kv":           # two-column key/value rows
            for k, v in el[1]:
                ax.text(0.105, y, k, fontsize=9.2, va="top", color=NAVY, family="monospace")
                w = textwrap.fill(v, width=66)
                ax.text(0.40, y, w, fontsize=9.2, va="top", color="black")
                y -= 0.0150 * (w.count("\n") + 1) + 0.004
            y -= 0.004
        elif kind == "img":          # embedded image: (path, x, y0, w, h)
            p = Path(el[1])
            if p.exists():
                axi = fig.add_axes([el[2], el[3], el[4], el[5]])
                axi.imshow(mpimg.imread(p)); axi.axis("off")
        elif kind == "s":            # vertical spacer
            y -= el[1]

    ax.text(0.5, 0.022, f"DESI × Planck Cross-Correlation Program  —  page {page_no} of {total}",
            ha="center", fontsize=8, color=GREY)
    pdf.savefig(fig); plt.close(fig)


def box(ax, x, y, w, h, text, fc="#eaf0fb", ec=NAVY, fs=8.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="black", wrap=True)


def arrow(ax, p0, p1, color=NAVY):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.3, color=color, shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- flowchart page
def flowchart_page(pdf, page_no, total):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, "Pipeline Flowchart", ha="center", fontsize=16.5, fontweight="bold")
    ax.plot([0.08, 0.92], [0.915, 0.915], color="#333", lw=1.0)

    # inputs
    box(ax, 0.06, 0.83, 0.40, 0.055, "DESI DR1/DR2 catalogs\n(data + randoms, NGC+SGC)", fc="#fdeee0")
    box(ax, 0.54, 0.83, 0.40, 0.055, "Planck maps\nT (SMICA/Cmdr/SEVEM) + lensing κ", fc="#fdeee0")

    # maps row
    box(ax, 0.06, 0.73, 0.40, 0.055, "desi_build_maps.py\nδ_g(z bin), LSS mask", fc="#eaf0fb")
    box(ax, 0.54, 0.73, 0.40, 0.055, "planck_prepare_maps.py\nT, κ, DESI∩Planck joint mask", fc="#eaf0fb")

    # n(z) + templates
    box(ax, 0.06, 0.63, 0.40, 0.055, "make_perbin_nz.py\nweighted n(z) per bin", fc="#eaf0fb")
    box(ax, 0.54, 0.63, 0.40, 0.055, "build_isw_templates (CAMB)\ngT=TxW, gg=WxW, κg=PxW, TT", fc="#eaf0fb")

    # cross spectra
    box(ax, 0.06, 0.52, 0.88, 0.058,
        "compute_crosscls.py  (NaMaster / MASTER pseudo-Cℓ)\n"
        "δ_g × T → gT      δ_g × κ → κg      δ_g auto → gg", fc="#e7f5ec", bold=True)

    # covariance
    box(ax, 0.06, 0.42, 0.40, 0.055, "sim_covariance.py\nMonte-Carlo bandpower cov", fc="#eaf0fb")
    box(ax, 0.54, 0.42, 0.40, 0.055, "run_nulls.py\ncurl · comp_diff · ngc_sgc", fc="#eaf0fb")

    # stage A / B
    box(ax, 0.06, 0.30, 0.40, 0.075, "STAGE A:  fit_bias_growth\n{gg, κg}  →  b(z), σ8\n(κg ~30–50σ pins bias)",
        fc="#f3e9fb", bold=True)
    box(ax, 0.54, 0.30, 0.40, 0.075, "STAGE B:  fit_isw_amplitudes\ngT with b(z) fixed  →  A_ISW(z)",
        fc="#f3e9fb", bold=True)

    # combine
    box(ax, 0.20, 0.18, 0.60, 0.058, "combine_aisw.py\nA_ISW(z) curve + 3×2pt (gg, κg, gT) summary", fc="#e7f5ec", bold=True)

    # orchestrator note
    box(ax, 0.20, 0.075, 0.60, 0.05, "run_phase1_desi.py — per-tracer orchestrator (BGS·LRG·ELG·QSO)", fc="#f5f5f5")

    a = lambda p0, p1: arrow(ax, p0, p1)
    a((0.26, 0.83), (0.26, 0.785)); a((0.74, 0.83), (0.74, 0.785))
    a((0.26, 0.73), (0.26, 0.685)); a((0.74, 0.73), (0.74, 0.685))
    a((0.26, 0.63), (0.40, 0.578)); a((0.74, 0.63), (0.60, 0.578))
    a((0.50, 0.52), (0.26, 0.475)); a((0.50, 0.52), (0.74, 0.475))
    a((0.26, 0.42), (0.26, 0.375)); a((0.74, 0.42), (0.74, 0.375))
    a((0.46, 0.34), (0.54, 0.34))  # A feeds B (bias pinned)
    a((0.26, 0.30), (0.40, 0.238)); a((0.74, 0.30), (0.60, 0.238))
    a((0.50, 0.18), (0.50, 0.125))

    ax.text(0.5, 0.022, f"DESI × Planck Cross-Correlation Program  —  page {page_no} of {total}",
            ha="center", fontsize=8, color=GREY)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- build document
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="DESI_xcorr_Program.pdf")
    ap.add_argument("--plots-dir", default="results/plots")
    args = ap.parse_args()
    TOT = 13
    pd = args.plots_dir

    with PdfPages(args.out) as pdf:
        # 1. Title
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.5, 0.74, "DESI × Planck", ha="center", fontsize=30, fontweight="bold")
        ax.text(0.5, 0.685, "Cross-Correlation Program", ha="center", fontsize=24, fontweight="bold", color=NAVY)
        ax.text(0.5, 0.63, "CMB lensing (κg) + Integrated Sachs–Wolfe (gT) tomography",
                ha="center", fontsize=13, color=GREY, style="italic")
        ax.plot([0.25, 0.75], [0.60, 0.60], color="#333", lw=1)
        ax.text(0.5, 0.55,
                "Measure DESI galaxies × Planck (temperature and lensing) to pin galaxy bias and\n"
                "growth from κg, then infer the redshift evolution of the gravitational-potential\n"
                "decay A_ISW(z) — a direct, distance-independent probe of dark energy.",
                ha="center", fontsize=11)
        ax.text(0.5, 0.30, "Design, mathematics, pipeline, and plan", ha="center", fontsize=12, fontweight="bold")
        ax.text(0.5, 0.07, "Generated by scripts/make_project_doc.py   ·   env: clankv4_dev   ·   2026",
                ha="center", fontsize=8.5, color=GREY)
        pdf.savefig(fig); plt.close(fig)

        # 2. Purpose & change of direction
        render_page(pdf, "Purpose & Change of Direction", [
            ("h", "Original goal"),
            ("p", "Measure tomographic ISW: DESI galaxies × Planck temperature → A_ISW(z), the redshift "
                  "evolution of the decaying gravitational potential. ISW is the unique linear probe of "
                  "∂Φ/∂t and hence of dark energy / modified gravity at the largest scales."),
            ("h", "Why broaden the scope"),
            ("p", "Validating the ISW pipeline on LRG showed (a) the theory engine had normalization bugs that "
                  "had to be fixed, and (b) ISW alone is intrinsically low signal-to-noise — DR1 forecast ≈ 2σ. "
                  "The same galaxy maps and machinery, crossed with Planck CMB lensing κ, give a 30–50σ signal "
                  "that pins galaxy bias and growth — turning a marginal measurement into a robust program."),
            ("h", "The program: three observables, one pipeline"),
            ("b", ["gg — galaxy clustering auto: bias × growth.",
                   "κg — CMB-lensing × galaxy: high S/N; pins b(z) and σ8 / S8.",
                   "gT — ISW × galaxy: the dark-energy ∂Φ/∂t handle (and modified-gravity test)."]),
            ("h", "Two-stage logic"),
            ("p", "Stage A fits {gg, κg} → solve b(z) and σ8. Stage B fixes b(z) and fits the one-parameter "
                  "A_ISW(z) cleanly. κg and ISW probe dark energy through different physics — growth vs "
                  "potential-decay — and break different degeneracies, so the combination exceeds the sum."),
        ], 2, TOT)

        # 3. Repository & environment
        render_page(pdf, "Repository Structure & Environment", [
            ("h", "Environment (pinned)"),
            ("p", "All scripts run in the conda env clankv4_dev (healpy + pymaster/NaMaster + clank-v4 + CAMB). "
                  "Theory: gg/gT/κg templates from CAMB source windows; clank-v4 used for the bias-calibration "
                  "auto theory. Large data (DESI catalogs, Planck maps) live outside the repo."),
            ("h", "Layout"),
            ("kv", [("configs/desi.yml", "paths, tomographic bins, estimator settings, fiducial cosmology"),
                    ("scripts/", "pipeline (see the script map page)"),
                    ("templates/isw/", "per-bin n(z) and CAMB gT/gg/κg/TT templates"),
                    ("results/", "bias priors, A_ISW tables, sim covariances, plots"),
                    ("Data/DESI/<TRACER>/", "clustering data + randoms (NGC+SGC)"),
                    ("Data/Planck/", "SMICA/Commander/SEVEM T, union mask, lensing κ (PR4)"),
                    ("ISWresults/", "maps/ masks/ spectra/ (versioned by NSIDE, ℓ-max, bin)")]),
            ("h", "Conventions"),
            ("b", ["Per-tracer, per-bin products; outputs versioned by NSIDE / ℓmax / bin index.",
                   "Same commands run locally (nside 512, LRG) and on Pawsey (nside 1024, all tracers).",
                   "δ_g = (D − αR)/(αR) with full systematic weights; masked pixels carry hp.UNSEEN."]),
        ], 3, TOT)

        # 4. Flowchart
        flowchart_page(pdf, 4, TOT)

        # 5. Theory I — fields and angular spectra
        render_page(pdf, "Theory I — Fields & Angular Power Spectra", [
            ("p", "Each projected field is a line-of-sight integral of a 3-D field against a radial kernel W(χ). "
                  "In the Limber approximation the cross-spectrum of two tracers a, b is:"),
            ("eq", r"C_\ell^{ab} = \int \frac{d\chi}{\chi^2}\, W_a(\chi)\,W_b(\chi)\, P\left(k=\frac{\ell+1/2}{\chi},\,z(\chi)\right)"),
            ("p", "with the matter power spectrum P(k,z) and comoving distance χ(z). Units must be consistent "
                  "(χ in Mpc ⇒ P in Mpc³): a mismatch here was one of the clank-v4 bugs."),
            ("h", "Galaxy number counts"),
            ("eq", r"W_g(\chi) = b(z)\,n(z)\,\frac{dz}{d\chi},\qquad \int n(z)\,dz = 1"),
            ("p", "giving the auto C_ℓ^{gg} ∝ b² P (clustering) plus shot noise N = 1/n̄. The galaxy bias b(z) "
                  "connects the observed galaxy overdensity to the matter field, δ_g = b δ_m."),
            ("h", "The three spectra we use"),
            ("b", ["C_ℓ^{gg}: galaxy auto — measures b²σ8² (+ shot noise).",
                   "C_ℓ^{κg}: lensing × galaxy — measures b σ8² at high S/N.",
                   "C_ℓ^{gT}: ISW × galaxy — measures the potential decay (∝ A_ISW)."]),
            ("p", "Because C^{gg} ∝ b²σ8² and C^{κg} ∝ b σ8², the pair {gg, κg} separates b(z) and σ8 — the basis "
                  "of Stage A."),
        ], 5, TOT)

        # 6. Theory II — ISW
        render_page(pdf, "Theory II — Integrated Sachs–Wolfe (gT)", [
            ("p", "As CMB photons cross evolving potentials they gain/lose energy. In matter domination the "
                  "potential is static (Φ ∝ (1+z)D = const) and there is no effect; once dark energy dominates "
                  "the potential decays and imprints a temperature shift:"),
            ("eq", r"\left(\frac{\Delta T}{T}\right)_{\mathrm{ISW}} = \frac{2}{c^2}\int \frac{\partial \Phi}{\partial \eta}\, d\chi"),
            ("p", "Cross-correlating with galaxies (Limber, linear) gives the working expression used for the "
                  "templates:"),
            ("eq", r"C_\ell^{gT} = -\,\frac{3\,\Omega_m H_0^2\,T_{\mathrm{CMB}}}{(\ell+1/2)^2\,c^2}\int dz\, b\,n(z)\,D(z)\,\frac{H(z)}{c}\,\frac{d[(1+z)D]}{dz}\, P(k,0)"),
            ("h", "Redshift dependence — the whole point"),
            ("b", ["Kernel d[(1+z)D]/dz ≈ 0 for z ≳ 2 (matter era) → no ISW.",
                   "Peaks at z ≈ 0.3–0.6 as Λ takes over → signal strongest at low z.",
                   "A_ISW(z) tests when dark energy switched on; modified gravity predicts a different curve."]),
            ("p", "Practical note: ISW lives at ℓ ≲ 60 where the Limber approximation is poor, so the gT (and κg) "
                  "templates are computed with CAMB source windows (full Boltzmann TxW / PxW), not Limber."),
        ], 6, TOT)

        # 7. Theory III — lensing
        render_page(pdf, "Theory III — CMB Lensing (κg) & Bias/Growth", [
            ("p", "CMB lensing convergence κ is sourced by all matter between us and last scattering (χ∗); its "
                  "radial kernel is broad and peaks at z ~ 1–2:"),
            ("eq", r"W_\kappa(\chi) = \frac{3\,\Omega_m H_0^2}{2 c^2}\,\frac{\chi}{a(\chi)}\,\frac{\chi_\ast-\chi}{\chi_\ast}"),
            ("eq", r"C_\ell^{\kappa g} = \int \frac{d\chi}{\chi^2}\, W_\kappa(\chi)\,W_g(\chi)\, P\left(k,\,z\right)"),
            ("h", "Why κg is ~30–50σ but gT is ~2σ"),
            ("b", ["κ traces matter directly → high correlation with galaxies (r ~ 0.5–0.8); T is dominated by the "
                   "primordial CMB, so the ISW part is buried.",
                   "κg has signal to ℓ ~ 1000s (thousands of modes); ISW only to ℓ ~ 60 (few modes). S/N ∝ √Nmodes.",
                   "Lensing is a large, coherent deflection; ISW is a faint µK-level large-scale ripple."]),
            ("h", "Separating bias and growth (Stage A)"),
            ("eq", r"C^{gg}\propto b^2\sigma_8^2,\quad C^{\kappa g}\propto b\,\sigma_8^2 \;\Rightarrow\; b,\ \sigma_8 \ \mathrm{separable}"),
            ("p", "Fixing b(z) from {gg, κg} removes the dominant ISW systematic (A_ISW is degenerate with bias) "
                  "and yields an independent growth / S8 measurement as a bonus."),
        ], 7, TOT)

        # 8. Estimator & covariance
        render_page(pdf, "Estimator & Covariance", [
            ("h", "Pseudo-Cℓ (MASTER) on a cut, apodized sky"),
            ("p", "Masking couples multipoles. NaMaster builds fields on the apodized DESI∩Planck joint mask and "
                  "deconvolves the mode-coupling matrix to return unbiased binned bandpowers (Δℓ = 10, ℓ ≲ 150). "
                  "Bad/empty pixels (hp.UNSEEN) are folded into the mask and zeroed in the map."),
            ("h", "Amplitude (GLS)"),
            ("eq", r"\hat A = \frac{t^{T} C^{-1} d}{t^{T} C^{-1} t},\qquad \sigma_A = \left(t^{T} C^{-1} t\right)^{-1/2}"),
            ("p", "with data bandpowers d, theory template t = C_ℓ^{gT,fid}, and bandpower covariance C. Stacking "
                  "tomographic bins uses the full bin-bin covariance (bins share one CMB → correlated)."),
            ("h", "Covariance from simulations"),
            ("p", "The analytic Gaussian covariance is optimistic (it ignores mask mode-coupling and bin-bin "
                  "correlations). We draw Gaussian CMB realisations, cross each with all galaxy bins, and take the "
                  "sample covariance; the inverse is Hartlap-corrected:"),
            ("eq", r"C^{-1}_{\mathrm{unbiased}} = \frac{N_{\mathrm{sim}}-p-2}{N_{\mathrm{sim}}-1}\,C^{-1}"),
            ("p", "for p data points and N_sim realisations. This same covariance is used in the null-test χ²."),
        ], 8, TOT)

        # 9. Script map
        render_page(pdf, "Scripts & How They Fit Together", [
            ("kv", [
                ("desi_build_maps.py", "catalogs → δ_g maps (per bin, per cap), LSS mask"),
                ("planck_prepare_maps.py", "T/κ maps to NSIDE; DESI∩Planck joint mask"),
                ("make_perbin_nz.py", "weighted per-bin n(z) from the clustering catalog"),
                ("build_isw_templates_clank.py", "CAMB source-window gT/gg/(κg)/TT templates per bin"),
                ("prepare_kappa.py", "Planck κ_lm → rotated κ map + DESI∩lensing joint mask"),
                ("compute_crosscls.py", "NaMaster δ_g × {T, κ} and gg → bandpowers (--cmb-field)"),
                ("validate_kappa_template.py", "κg vs CAMB-Limber theory + literature bias check"),
                ("fit_bias_growth.py", "joint {gg, κg} → κ-pinned b(z), σ8 (Stage A)"),
                ("build_isw_templates_clank.py", "CAMB source-window gT/gg templates per bin"),
                ("fit_isw_amplitudes.py", "GLS A_ISW(z) with bias pinned (Stage B)"),
                ("sim_covariance.py", "Monte-Carlo bandpower covariance + refit"),
                ("run_nulls.py", "curl, comp_diff, ngc_sgc, ell_range (sims cov)"),
                ("combine_aisw.py · make_3x2pt_summary.py", "A_ISW(z) curve · gg+κg+gT capstone"),
                ("grab_and_build.sh · run_xcorr_all.sh", "download+build maps · full multi-tracer chain"),
            ]),
            ("p", "Per-tracer outputs feed combine_aisw; grab_and_build.sh builds any tracer's maps and "
                  "run_xcorr_all.sh runs the whole gg+κg+gT chain across all four tracers to the combine."),
        ], 9, TOT)

        # 10. clank-v4
        render_page(pdf, "Theory Engine: clank-v4 Fixes & Validation", [
            ("p", "clank-v4 is the in-house theory code. Validating it against CAMB during this work uncovered "
                  "three bugs; the strategy throughout was to cross-check every template against an independent "
                  "CAMB calculation before trusting it."),
            ("h", "Bugs found"),
            ("b", ["Ωm collapsed to Ωb only (a parameter-name mismatch) → E(z)/H(z)/dZdχ wrong while χ(z) "
                   "(from CAMB) looked fine. FIXED in clank.",
                   "Limber/full-z integral used P in (Mpc/h)³ with χ in Mpc → every Cℓ low by h³. FIXED in clank.",
                   "ISW kernel wrong (sign, ~10× amplitude, ℓ-shape) → A_ISW suppressed ~10×. BYPASSED: gT (and "
                   "gg, κg) templates now come from CAMB source windows."]),
            ("h", "Validation results"),
            ("b", ["After fixes, clank C_ℓ^{gg} matches CAMB-Limber to ~4% with no z-trend (was 0.205 = h⁴).",
                   "gT template matches CAMB full-Boltzmann TxW to 1e-4.",
                   "Recovered LRG bias b ≈ 1.9–2.4 (textbook), vs the spurious ~4.6–7.7 before the fixes."]),
            ("h", "A separate, decisive data-pipeline bug"),
            ("p", "Planck maps are native Galactic; DESI maps are pixelised in Equatorial. The map prep "
                  "relabeled coord=C without rotating, so every CMB×galaxy cross was frame-mismatched — which "
                  "for the faint ISW just looked like a null. Caught by κg (rotating κ to Equatorial gives a "
                  "~6× stronger galaxy cross). Fixing it (rotate G→C) turned A_ISW from a frame-scrambled "
                  "−0.45±0.87 into 0.94±0.67, on ΛCDM — i.e. it revealed the signal."),
        ], 10, TOT)

        # 11. Current status & results (with figures)
        render_page(pdf, "Current Status — LRG (two-stage, full sky)", [
            ("p", "The LRG analysis runs end-to-end (catalog → curve) on the full footprint (fsky ≈ 0.22). "
                  "Stage A pins the bias from CMB lensing; Stage B fits the ISW with that bias fixed."),
            ("b", ["κg detected at ~14.5σ → bias pinned: b(z) = 2.07, 2.06, 2.31 (matches clustering + "
                   "the DESI LRG literature ~2.0).",
                   "A_ISW = 0.94 ± 0.67 (bias pinned, sims covariance) — consistent with ΛCDM (0.1σ), "
                   "1.4σ from zero.",
                   "All CMB nulls pass; templates CAMB-validated; covariance from 300 sims.",
                   "Frame fix (Galactic→Equatorial) revealed the signal: pre-fix A_ISW was −0.45±0.87, "
                   "a frame-scrambled null."]),
            ("img", f"{pd}/LRG_3x2pt.png", 0.06, 0.30, 0.88, 0.24),
            ("s", 0.27),
            ("p", "LRG 3×2pt: gg (clustering), κg (CMB lensing — points on theory, the high-S/N anchor), "
                  "gT (ISW — small theory, marginal signal), measured vs theory per z-bin."),
            ("b", ["4-tracer A_ISW(z) (BGS+LRG+ELG+QSO, z≈0.1–2): joint A = 0.81 ± 0.62 (ΛCDM 0.3σ, "
                   "1.3σ from 0); naive 1.25 ± 0.49 (2.6σ, optimistic).",
                   "κ×T (CMB-internal ISW, no galaxies): A_κT = 1.50 ± 0.47 (~4.6σ) — independent confirmation."]),
        ], 11, TOT)

        # 12. Forecast & strategy
        render_page(pdf, "Forecast & Strategy", [
            ("h", "Honest CAMB Fisher forecast (DR1)"),
            ("kv", [("BGS (z≈0.1)", "S/N ≈ 0.2     (too low-z / thin)"),
                    ("LRG (z≈0.7)", "S/N ≈ 1.6"),
                    ("ELG (z≈1.1)", "S/N ≈ 1.1"),
                    ("QSO (z≈1.7)", "S/N ≈ 1.3"),
                    ("joint (4 tracers)", "S/N ≈ 2.1    (σ_A ≈ 0.48)")]),
            ("p", "Matches the realised LRG sims-cov error and the published imaging benchmark (Hang 2021: "
                  "A_ISW = 0.98 ± 0.35 at fsky ≈ 0.4) once sky is matched."),
            ("h", "Context"),
            ("b", ["ISW has a hard ceiling of ~7–8σ (cosmic-variance limited at low ℓ) for any survey.",
                   "More sky is the dominant lever: DR1 ~2σ → full DESI footprint ~3–4σ.",
                   "κg (~30–50σ) is the high-S/N partner: pins bias + growth, makes the ISW result trustworthy.",
                   "Published DESI ISW work is imaging-based and non-tomographic — a spectroscopic, tomographic "
                   "A_ISW(z) would be a genuine first."]),
        ], 12, TOT)

        # 13. The plan
        render_page(pdf, "Status & Plan (κg + ISW Program)", [
            ("h", "Done — LRG, end to end"),
            ("b", ["Stage 0: Planck PR3 lensing (COM_Lensing_4096, MV) + κ prep (prepare_kappa.py). ✓",
                   "Stage A: κg spectra (14.5σ), validate_kappa_template, fit_bias_growth → b(z) pinned. ✓",
                   "Stage B: CAMB templates → A_ISW = 0.94±0.67 (ΛCDM) + sims cov + nulls pass. ✓",
                   "3×2pt capstone (gg+κg+gT) for LRG. ✓"]),
            ("h", "Done — all four tracers + extensions"),
            ("b", ["BGS/LRG/ELG/QSO full gg+κg+gT chain (run_xcorr_all.sh); 4-tracer A_ISW(z), z≈0.1–2.",
                   "Joint A with the cross-tracer amplitude covariance; κ×T CMB-internal ISW cross-check.",
                   "ELG bins rebinned to data; pixel window deconvolved at the spectrum stage."]),
            ("h", "Remaining / refinements"),
            ("b", ["Smooth A_ISW(z) (2-param) evolution fit on the joint amplitude covariance.",
                   "κ×T over the full Planck lensing footprint (fsky 0.67) for higher S/N.",
                   "DESI DR2 (not yet public) — more sky is the main S/N lever; then a Pawsey nside-1024 run.",
                   "Fix clank's ISW kernel at source if clank is reused (currently bypassed via CAMB)."]),
            ("h", "Execution order"),
            ("p", "0 (data) → A (κg, b/σ8) → B (ISW pinned) → nulls → 4-tracer combine. Same config-driven "
                  "commands scale from local LRG (nside 512) to the full survey."),
        ], 13, TOT)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
