#!/usr/bin/env python3
"""
Build per-tomographic-bin n(z) tables for the ISW theory templates.

For each tracer and each bin in cfg["desi"]["bins"][TRACER], write a 2-column
text file  `z  nz`  to  {outdir}/{TRACER}_z{ibin}_nz.txt  (ibin = 1-based, matching
the delta_{TRACER}_z{ibin} maps and gT_{TRACER}_z{ibin} spectra).

Two modes
---------
catalog (default): weighted histogram of the DESI clustering data catalog.
    n(z) ∝ Σ_g WEIGHT_g  per dz, using the *same* Z column and weight composition
    as desi_build_maps.py (reused helpers), so the template kernel matches the maps.

--from-nztable: slice the pre-combined DESI table {TRACER}_NGCplusSGC_nz.txt at the
    bin edges. Catalog-free cross-check / fallback (the DESI tables are already
    weighted server-side).

These per-bin n(z) files are consumed by build_isw_templates_clank.py to produce
per-bin gT/gg theory templates → genuine tomographic A_ISW(z).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

# Reuse the catalog-reading + weighting helpers from the map builder so the
# template n(z) is constructed identically to the overdensity maps.
from desi_build_maps import (
    read_columns,
    best_redshift_name,
    build_weight,
    find_lss_files,
)

LOGGER = logging.getLogger("make_perbin_nz")

# Columns we may need from the data catalog (superset; read_columns keeps what exists).
NEED_COLS = ["RA", "DEC", "WEIGHT", "WEIGHT_COMP", "WEIGHT_SYS", "WEIGHT_ZFAIL",
             "WEIGHT_IMAGING", "WEIGHT_SYSTOT", "WEIGHT_FKP", "FRAC_TLOBS", "FTILE",
             "Z", "Z_NOT4CLUS", "Z_noqso", "Z_QA"]


def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tracers", nargs="+", default=None,
                   help="Subset to process (default: all tracers in cfg['desi']['bins'])")
    p.add_argument("--cap", choices=["NGC", "SGC", "ANY"], default="ANY",
                   help="Sky cap(s) to combine (catalog mode)")
    p.add_argument("--outdir", default="templates/isw/nz",
                   help="Where to write {TRACER}_z{ibin}_nz.txt")
    p.add_argument("--dz", type=float, default=0.01,
                   help="Histogram bin width in z within each tomographic bin")
    p.add_argument("--from-nztable", action="store_true",
                   help="Slice {TRACER}_NGCplusSGC_nz.txt instead of reading the catalog")
    p.add_argument("--nztable-dir", default=".",
                   help="Directory holding {TRACER}_NGCplusSGC_nz.txt (--from-nztable mode)")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def write_nz(path: Path, z: np.ndarray, nz: np.ndarray, header: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.c_[z, nz], header=header, fmt=["%.5f", "%.10e"])
    LOGGER.info("  wrote %s  (%d z-points, Σnz=%.3e)", path, z.size, float(np.sum(nz)))


# ---------- catalog mode ----------

def load_catalog_z_w(cfg: dict, tracer: str, cap: str):
    """Return (z, weight) arrays concatenated over the tracer's data catalogs."""
    files = find_lss_files(cfg, tracer, cap)  # {"data": [...], "randoms": [...]}
    zs, ws = [], []
    for f in files["data"]:
        cols = read_columns(f, NEED_COLS)
        zname = best_redshift_name(set(cols.keys()))
        if zname is None:
            raise RuntimeError(f"No redshift column in {f}")
        w = build_weight(cols, for_random=False)
        w = np.where(w < 0, 0.0, w)
        z = np.asarray(cols[zname], dtype=np.float64)
        good = np.isfinite(z) & np.isfinite(w)
        zs.append(z[good]); ws.append(w[good])
    return np.concatenate(zs), np.concatenate(ws)


def perbin_nz_from_catalog(z_all, w_all, zmin, zmax, dz):
    """Weighted histogram of Z in [zmin, zmax) → (z_centers, nz)."""
    nb = max(1, int(round((zmax - zmin) / dz)))
    edges = np.linspace(zmin, zmax, nb + 1)
    sel = (z_all >= zmin) & (z_all < zmax)
    nz, _ = np.histogram(z_all[sel], bins=edges, weights=w_all[sel])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, nz


# ---------- nztable mode ----------

def perbin_nz_from_table(table_path: Path, zmin, zmax):
    """Slice rows of a DESI combined nz.txt (zmid zlow zhigh nz ...) into [zmin,zmax)."""
    arr = np.loadtxt(table_path)
    zmid, nz = arr[:, 0], arr[:, 3]
    sel = (zmid >= zmin) & (zmid < zmax)
    if sel.sum() == 0:
        raise RuntimeError(f"No n(z) rows of {table_path.name} fall in [{zmin},{zmax})")
    return zmid[sel], nz[sel]


def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    outdir = Path(args.outdir)

    bins_cfg = cfg["desi"]["bins"]
    tracers = args.tracers or list(bins_cfg.keys())

    for tracer in tracers:
        bins = bins_cfg[tracer]
        LOGGER.info("Tracer %s: %d bins", tracer, len(bins))

        z_all = w_all = None
        if not args.from_nztable:
            z_all, w_all = load_catalog_z_w(cfg, tracer, args.cap)
            LOGGER.info("  catalog: %d objects (weighted sum %.3e)",
                        z_all.size, float(w_all.sum()))

        for idx, (zmin, zmax) in enumerate(bins):
            ibin = idx + 1
            if args.from_nztable:
                table = Path(args.nztable_dir) / f"{tracer}_NGCplusSGC_nz.txt"
                z, nz = perbin_nz_from_table(table, zmin, zmax)
                src = f"sliced {table.name}"
            else:
                z, nz = perbin_nz_from_catalog(z_all, w_all, zmin, zmax, args.dz)
                src = f"catalog weighted-hist dz={args.dz}"
            hdr = f"{tracer} z{ibin} [{zmin},{zmax})  source: {src}\nz  nz"
            write_nz(outdir / f"{tracer}_z{ibin}_nz.txt", z, nz, hdr)


if __name__ == "__main__":
    main()
