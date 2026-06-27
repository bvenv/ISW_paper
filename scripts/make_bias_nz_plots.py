#!/usr/bin/env python3
"""Refresh the b(z) and n(z) summary plots from current data (kappa-pinned bias + catalog n(z))."""
import argparse, glob, json, re
from pathlib import Path
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

COL = {"BGS": "#d62728", "LRG": "#1f77b4", "ELG": "#2ca02c", "QSO": "#9467bd"}

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--bias-glob", default="results/bias/*_bias_kappa.json")
ap.add_argument("--nz-dir", default="templates/isw/nz")
ap.add_argument("--outdir", default="results/plots")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
Path(a.outdir).mkdir(parents=True, exist_ok=True)

# ---- b(z) ----
fig, ax = plt.subplots(figsize=(7, 4.4))
for f in sorted(glob.glob(a.bias_glob)):
    T = re.search(r"/([A-Z]+)_bias_kappa", f).group(1)
    pr = json.load(open(f))["priors"]
    z = [0.5 * (p["zmin"] + p["zmax"]) for p in pr]
    b = [p["b"] for p in pr]; sb = [p["sigma_b"] for p in pr]
    ax.errorbar(z, b, yerr=sb, fmt="o-", capsize=3, color=COL.get(T, "k"), label=T)
ax.set_xlabel(r"$\bar z_{\rm bin}$"); ax.set_ylabel("linear bias $b$")
ax.set_title("DESI galaxy bias vs redshift (κ-pinned)")
ax.grid(ls=":", alpha=.4); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(Path(a.outdir) / "desi_bias_vs_z.png", dpi=150); plt.close(fig)

# ---- n(z) per bin ----
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for T in cfg["desi"]["bins"]:
    for ib in range(1, len(cfg["desi"]["bins"][T]) + 1):
        p = Path(a.nz_dir) / f"{T}_z{ib}_nz.txt"
        if not p.exists():
            continue
        z, nz = np.loadtxt(p, unpack=True)
        ax.plot(z, nz / np.trapz(nz, z), color=COL.get(T, "k"),
                label=T if ib == 1 else None, alpha=0.8)
ax.set_xlabel("redshift $z$"); ax.set_ylabel("normalised $n(z)$ per bin")
ax.set_title("DESI per-bin n(z) (catalog-weighted)")
ax.grid(ls=":", alpha=.4); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(Path(a.outdir) / "desi_nz_per_bin.png", dpi=150); plt.close(fig)
print("Refreshed results/plots/desi_bias_vs_z.png + desi_nz_per_bin.png")
