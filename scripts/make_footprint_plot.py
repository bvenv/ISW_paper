#!/usr/bin/env python3
"""DESI × Planck footprint figure: the joint analysis masks (T and κ) in Mollweide."""
import argparse
from pathlib import Path
import numpy as np
import yaml
import healpy as hp
import matplotlib; matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--nside", type=int, default=512)
ap.add_argument("--out", default="results/plots/footprint.png")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
masks = Path(cfg["paths"]["masks"])

jt = masks / f"planck_desi_joint_nside{a.nside}.fits.gz"          # DESI ∩ Planck-T
jk = masks / f"planck_desi_kappa_joint_nside{a.nside}.fits.gz"    # DESI ∩ Planck-lensing

fig = plt.figure(figsize=(8, 8.4))
for i, (path, title) in enumerate([(jt, r"DESI $\cap$ Planck-T (ISW)"),
                                    (jk, r"DESI $\cap$ Planck lensing ($\kappa$)")]):
    m = hp.read_map(str(path)) if path.exists() else None
    if m is None:
        continue
    fsky = float(np.mean(m > 0))   # combined footprint (per-tracer values are in Table 2)
    hp.mollview(m, sub=(2, 1, i + 1), title=f"{title}   ($f_{{\\rm sky}}={fsky:.2f}$)",
                cmap="viridis", cbar=False, min=0, max=1,
                margins=(0.02, 0.04, 0.02, 0.04))
    hp.graticule(dpar=30, dmer=60, alpha=0.3)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"wrote {a.out}")
