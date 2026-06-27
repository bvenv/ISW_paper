#!/usr/bin/env python3
"""
Build DESI tomographic HEALPix overdensity maps per tracer/bin using LSScats (+randoms).

Outputs per bin:
  maps/desi/{tracer}/nside{NSIDE}/delta_{tracer}_z{ibin}.fits.gz
Also writes (per tracer):
  masks/desi_lssmask_{tracer}_nside{NSIDE}.fits.gz

Notes
- Uses clustering data/randoms and applies the *same* z-cuts to both.
- Overdensity: delta = (D - alpha R) / (alpha R), alpha = sum(w_data)/sum(w_rand).
- Requires random catalogs to have a redshift column (e.g., 'Z'); will error if absent.
"""
import argparse
import logging
from pathlib import Path
import json
import sys
import glob

# I/O backends (fitsio preferred; fall back to astropy)
try:
    import fitsio
    HAS_FITSIO = True
except Exception:
    HAS_FITSIO = False
    from astropy.io import fits  # type: ignore

import numpy as np
import healpy as hp
import yaml

LOGGER = logging.getLogger("desi_build_maps")


# ---------- CLI / config ----------
def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="Path to YAML config (configs/desi.yml)")
    p.add_argument("--tracers", nargs="+", default=["BGS", "LRG", "ELG", "QSO"], help="Tracers to build")
    p.add_argument("--cap", choices=["NGC", "SGC", "ANY"], default="ANY", help="Sky cap selection")
    p.add_argument("--nside", type=int, default=1024, help="HEALPix NSIDE for output maps")
    p.add_argument("--rmin-threshold", type=float, default=0.0, help="Minimum R[p] to keep pixel (0 keeps all R>0)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--qa", action="store_true", help="Write quick-look QA plots")
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("--save-counts", action="store_true",
               help="Also save weighted counts maps: D, R, and denom=alpha*R")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- helpers ----------
def _glob_one(patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted(set(out))


def find_lss_files(cfg: dict, tracer: str, cap: str):
    """Return dict with lists of data and random file paths for this tracer and cap."""
    base = cfg["paths"]["desi"]["lsscats"][tracer]
    # Typical file stems:
    #   {TRACER}_{CAP}_clustering.dat.fits
    #   {TRACER}_{CAP}_{i}_clustering.ran.fits   (i = 0..17 etc.)
    # Some tracers include qualifiers in names (e.g., ELG_LOPnotqso); match tracer prefix loosely.
    tracer_pat = tracer + "*"  # allows ELG_LOPnotqso, etc.

    caps = [cap] if cap in ("NGC", "SGC") else ["NGC", "SGC"]
    data_files, ran_files = [], []
    for c in caps:
        data_files += _glob_one([f"{base}/{tracer_pat}_{c}_clustering.dat.fits"])
        ran_files  += _glob_one([f"{base}/{tracer_pat}_{c}_*_clustering.ran.fits"])  # chunks
    if not data_files:
        raise FileNotFoundError(f"No clustering.dat.fits found for {tracer} in {base} (cap={cap})")
    if not ran_files:
        raise FileNotFoundError(f"No clustering.ran.fits found for {tracer} in {base} (cap={cap})")
    return {"data": data_files, "randoms": ran_files}


def read_columns(path, cols):
    """Read selected columns from a FITS table quickly."""
    if HAS_FITSIO:
        with fitsio.FITS(path, "r") as f:
            hdu = f[1]
            names = [n.upper() for n in hdu.get_colnames()]
            want = [c for c in cols if c.upper() in names]
            arrs = [hdu.read(columns=want)[w] for w in want]
            return {w.upper(): a for w, a in zip(want, arrs)}
    else:
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            names = [n.upper() for n in data.names]
            want = [c for c in cols if c.upper() in names]
            return {w.upper(): np.array(data[w]) for w in want}


def best_redshift_name(names):
    cand = ["Z", "Z_NOT4CLUS", "Z_NOT4CLUS", "Z_noqso", "Z_QA"]  # generous
    for c in cand:
        if c in names:
            return c
    return None


def build_weight(dict_cols, for_random=False):
    """Return a weight vector from available columns; prefer WEIGHT else compose."""
    names = set(dict_cols.keys())
    if "WEIGHT" in names:
        return np.asarray(dict_cols["WEIGHT"], dtype=np.float64)
    # Compose from pieces if present
    w = np.ones(len(next(iter(dict_cols.values()))), dtype=np.float64)
    for part in ["WEIGHT_COMP", "WEIGHT_SYS", "WEIGHT_ZFAIL", "WEIGHT_IMAGING", "WEIGHT_SYSTOT"]:
        if part in names:
            w *= np.asarray(dict_cols[part], dtype=np.float64)
    # Randoms sometimes carry FRAC_TLOBS or FTILE-like weights; multiply if present
    for part in ["FRAC_TLOBS", "FTILE", "WEIGHT_FKP"]:  # latter harmless for angular
        if part in names:
            w *= np.asarray(dict_cols[part], dtype=np.float64)
    return w


def to_pix(ra_deg, dec_deg, nside):
    theta = np.deg2rad(90.0 - dec_deg)
    phi   = np.deg2rad(ra_deg)
    return hp.ang2pix(nside, theta, phi, nest=False)


def accum_weighted_counts(pix, w, npix):
    bc = np.bincount(pix, weights=w, minlength=npix)
    # ensure float64
    return bc.astype(np.float64, copy=False)


def write_map(path, m, extra_hdr=None):
    hdr = {} if extra_hdr is None else dict(extra_hdr)
    hp.write_map(path, m.astype(np.float32), dtype=np.float32, nest=False,
                 coord="C", overwrite=True, fits_IDL=False, extra_header=list(hdr.items()))


# ---------- core per-bin workflow ----------
# ---------- core per-bin workflow ----------
def build_bin_map(tracer, cap, nside, zmin, zmax, data_files, ran_files,
                  out_map_path, out_mask_path, qa=False, rmin_threshold=0.0,
                  save_counts=False, ibin=None):

    LOGGER.info("Tracer=%s Cap=%s z=[%.3f, %.3f) NSIDE=%d", tracer, cap, zmin, zmax, nside)

    need_cols = ["RA","DEC","WEIGHT","WEIGHT_COMP","WEIGHT_SYS","WEIGHT_ZFAIL",
                 "WEIGHT_IMAGING","FRAC_TLOBS","FTILE","WEIGHT_FKP","Z","Z_NOT4CLUS","Z_NOT4CLUS"]

    # --- load & concat data ---
    ras, decs, zs, ws = [], [], [], []
    for f in data_files:
        cols = read_columns(f, need_cols)
        zname = best_redshift_name(set(cols.keys()))
        if zname is None:
            raise RuntimeError(f"No redshift column found in data file: {f}")
        w = build_weight(cols, for_random=False)
        if np.any(w < 0):
            nneg = int((w < 0).sum())
            LOGGER.warning("  Data file %s has %d negative weights; clipping to zero.", f, nneg)
            w = np.where(w < 0, 0.0, w)
        sel = (cols[zname] >= zmin) & (cols[zname] < zmax) & np.isfinite(cols["RA"]) & np.isfinite(cols["DEC"])
        ras.append(cols["RA"][sel]); decs.append(cols["DEC"][sel]); zs.append(cols[zname][sel]); ws.append(w[sel])

    ra_d = np.concatenate(ras); dec_d = np.concatenate(decs); w_d = np.concatenate(ws)
    nD = float(w_d.sum())
    LOGGER.info("  Data: %d objects (weighted sum %.3e) after cuts", ra_d.size, nD)

    # --- load & concat randoms ---
    ras, decs, zs, wr = [], [], [], []
    for f in ran_files:
        cols = read_columns(f, need_cols)
        zname = best_redshift_name(set(cols.keys()))
        if zname is None:
            raise RuntimeError(f"No redshift column found in random file: {f} (randoms must have a Z column for tomographic maps)")
        w = build_weight(cols, for_random=True)
        if np.any(w < 0):
            nneg = int((w < 0).sum())
            LOGGER.warning("  Random file %s has %d negative weights; clipping to zero.", f, nneg)
            w = np.where(w < 0, 0.0, w)
        sel = (cols[zname] >= zmin) & (cols[zname] < zmax) & np.isfinite(cols["RA"]) & np.isfinite(cols["DEC"])
        ras.append(cols["RA"][sel]); decs.append(cols["DEC"][sel]); wr.append(w[sel])

    ra_r = np.concatenate(ras); dec_r = np.concatenate(decs); w_r = np.concatenate(wr)
    nR = float(w_r.sum())
    LOGGER.info("  Randoms: %d objects (weighted sum %.3e) after cuts", ra_r.size, nR)

    # --- pixelize ---
    npix = hp.nside2npix(nside)
    pix_d = to_pix(ra_d, dec_d, nside)
    pix_r = to_pix(ra_r, dec_r, nside)
    D = accum_weighted_counts(pix_d, w_d, npix)
    R = accum_weighted_counts(pix_r, w_r, npix)

    # --- overdensity ---
    alpha = nD / nR if nR > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = alpha * R
        delta = (D - denom) / denom

    # mask: keep pixels with sufficient random support
    if rmin_threshold <= 0.0:
        keep = R > 0
    else:
        medR_seen = np.median(R[R > 0]) if np.any(R > 0) else 0.0
        keep = R >= (rmin_threshold * medR_seen)

    # build masked map
    m = np.full(npix, hp.UNSEEN, dtype=np.float32)
    m[keep] = delta[keep].astype(np.float32)

    # --- quick checks / metrics ---
    vals = delta[keep]
    fsky = float(keep.sum() / npix)
    mean_delta = float(np.nanmean(vals)) if vals.size else float("nan")
    std_delta  = float(np.nanstd(vals))  if vals.size else float("nan")
    frac_gt3   = float((np.abs(vals) > 3).mean()) if vals.size else 0.0
    frac_gt5   = float((np.abs(vals) > 5).mean()) if vals.size else 0.0

    R_seen = R[keep]
    if R_seen.size:
        p1, p5, p50, p95, p99 = np.percentile(R_seen, [1,5,50,95,99])
    else:
        p1 = p5 = p50 = p95 = p99 = 0.0
    LOGGER.info("  QA: fsky=%.3f  mean(δ)=%+.3e  std(δ)=%.3f  |δ|>3: %.3f  |δ|>5: %.3f",
                fsky, mean_delta, std_delta, frac_gt3, frac_gt5)
    LOGGER.debug("  QA: R percentiles [1,5,50,95,99]=[%.3f, %.3f, %.3f, %.3f, %.3f]",
                 p1, p5, p50, p95, p99)

    # --- write outputs ---
    out_map_path.parent.mkdir(parents=True, exist_ok=True)
    if out_mask_path is not None:
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = {
        "TRACER": tracer,
        "CAP": cap,
        "NSIDE": nside,
        "ZMIN": float(zmin),
        "ZMAX": float(zmax),
        "ALPHA": float(alpha),
        "NDATA": float(nD),
        "NRAND": float(nR),
        "FSKY": fsky,
        "MEANDEL": mean_delta,
        "STDDEL": std_delta,
        "FOUT3": frac_gt3,
        "FOUT5": frac_gt5,
        "R_P01": float(p1),
        "R_P05": float(p5),
        "R_P50": float(p50),
        "R_P95": float(p95),
        "R_P99": float(p99),
        "FORM": "(D - alpha R)/(alpha R)",
        "ORDERING": "RING",
    }
    write_map(out_map_path, m, hdr)

        # --- optional: save counts maps (D, R) and denom = alpha*R ---
    if save_counts:
        aux_dir = out_map_path.parent / "aux"
        aux_dir.mkdir(parents=True, exist_ok=True)

        # Nice matching tag for filenames
        if ibin is not None:
            tag = f"z{ibin}"
        else:
            tag = f"z{zmin:.2f}-{zmax:.2f}".replace(".", "p")

        # Build masked float32 maps (UNSEEN outside keep)
        D_map = np.full(npix, hp.UNSEEN, dtype=np.float32); D_map[keep] = D[keep].astype(np.float32)
        R_map = np.full(npix, hp.UNSEEN, dtype=np.float32); R_map[keep] = R[keep].astype(np.float32)
        denom = alpha * R
        Den_map = np.full(npix, hp.UNSEEN, dtype=np.float32); Den_map[keep] = denom[keep].astype(np.float32)

        hdr_aux = dict(hdr)  # reuse main header contents
        hdr_aux["MAPTYPE"] = "COUNTS"

        write_map(aux_dir / f"countsD_{tracer}_{tag}.fits.gz", D_map, hdr_aux | {"COUNTTYP": "DATA"})
        write_map(aux_dir / f"countsR_{tracer}_{tag}.fits.gz", R_map, hdr_aux | {"COUNTTYP": "RANDOM"})
        write_map(aux_dir / f"denom_{tracer}_{tag}.fits.gz",   Den_map, hdr_aux | {"COUNTTYP": "ALPHAxR"})
        LOGGER.info("  Wrote counts maps → %s", aux_dir)


    # lss mask (per-bin for now; union at higher level if desired)
    if out_mask_path is not None:
        lssmask = np.zeros(npix, dtype=np.uint8)
        lssmask[keep] = 1
        hp.write_map(out_mask_path, lssmask, dtype=np.uint8, nest=False, coord="C", overwrite=True, fits_IDL=False)

    LOGGER.info("  Wrote map → %s", out_map_path)
    if out_mask_path is not None:
        LOGGER.info("  Wrote LSS mask → %s", out_mask_path)

    # sidecar JSON with the same QA metrics (easy to grep later)
    try:
        qa_json = out_map_path.with_suffix("").with_name(
            f"qa_{tracer}_{cap}_z{zmin:.2f}-{zmax:.2f}_n{nside}.json"
        )
        qa_json.write_text(json.dumps({
            "tracer": tracer, "cap": cap, "nside": nside,
            "zmin": float(zmin), "zmax": float(zmax),
            "alpha": float(alpha), "ndata": float(nD), "nrand": float(nR),
            "fsky": fsky, "mean_delta": mean_delta, "std_delta": std_delta,
            "frac_abs_delta_gt3": frac_gt3, "frac_abs_delta_gt5": frac_gt5,
            "R_percentiles": {"p01": float(p1), "p05": float(p5), "p50": float(p50),
                              "p95": float(p95), "p99": float(p99)}
        }, indent=2))
    except Exception as e:
        LOGGER.warning("QA JSON write failed (ignored): %s", e)

    # --- quick QA plots (optional) ---
    if qa:
        try:
            import matplotlib.pyplot as plt
            import os
            # put QA plots alongside maps in a sibling 'plots' tree
            qadir = out_map_path.parents[3] / "plots" / "desi" / tracer  # e.g., <root>/plots/desi/LRG
            os.makedirs(qadir, exist_ok=True)

            import matplotlib
            matplotlib.use("Agg", force=True)

            # histogram of delta
            plt.figure(figsize=(7, 4))
            plt.hist(vals, bins=100, histtype="step")
            plt.xlabel(r"$\delta$"); plt.ylabel("pixels")
            plt.xlim((-1,10))
            plt.yscale('log')
            plt.title(f"{tracer}-{cap} z[{zmin},{zmax}) NSIDE={nside}")
            plt.tight_layout()
            plt.savefig(qadir / f"hist_delta_{tracer}_{cap}_z{zmin:.2f}-{zmax:.2f}_n{nside}.png", dpi=120)
            plt.close()

            # simple R histogram (diagnostic)
            if R_seen.size:
                plt.figure(figsize=(7, 4))
                plt.hist(np.log10(R_seen[R_seen>0]), bins=80, histtype="step")
                plt.xlabel(r"log10 R"); plt.ylabel("pixels")
                plt.title(f"R support {tracer}-{cap} z[{zmin},{zmax})")
                plt.tight_layout()
                plt.savefig(qadir / f"hist_R_{tracer}_{cap}_z{zmin:.2f}-{zmax:.2f}_n{nside}.png", dpi=120)
                plt.close()

            hp.mollview(m, title=f"{tracer}-{cap} δ z[{zmin},{zmax})", norm="hist")
            hp.graticule()
            plt.savefig(qadir / f"moll_delta_{tracer}_{cap}_z{zmin:.2f}-{zmax:.2f}_n{nside}.png", dpi=120)
            plt.close()
        except Exception as e:
            LOGGER.warning("QA plotting failed (ignored): %s", e)

    return keep




# ---------- main ----------
def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)

    nside = args.nside
    maps_root = Path(cfg["paths"]["maps"]) / "desi"
    masks_root = Path(cfg["paths"]["masks"])
    masks_root.mkdir(parents=True, exist_ok=True) 

    for tracer in args.tracers:
        # discover files for this tracer/cap
        files = find_lss_files(cfg, tracer, args.cap)
        data_files = files["data"]
        ran_files  = files["randoms"]

        # bins
        bins = cfg["desi"]["bins"][tracer]
        # Per-tracer mask file (we overwrite each bin; union at the end)
        tracer_mask_union = None
        tracer_keep_union = None

        # Tag per-cap maps (delta_{tracer}_cap{NGC,SGC}_z{ibin}) so they don't clobber the
        # combined ANY map and match run_nulls.delta_map_path for the ngc_sgc null test.
        cap_tag = f"_cap{args.cap}" if args.cap in ("NGC", "SGC") else ""
        for ibin, (zmin, zmax) in enumerate(bins, start=1):
            out_map = maps_root / tracer / f"nside{nside}" / f"delta_{tracer}{cap_tag}_z{ibin}.fits.gz"
            # we won't write a mask here
            keep = build_bin_map(
                tracer=tracer, cap=args.cap, nside=nside,
                zmin=float(zmin), zmax=float(zmax),
                data_files=data_files, ran_files=ran_files,
                out_map_path=out_map, out_mask_path=None,
                qa=args.qa, rmin_threshold=args.rmin_threshold,
                save_counts=args.save_counts, ibin=ibin
            )

            tracer_keep_union = keep if tracer_keep_union is None else (tracer_keep_union | keep)

        # after the bin loop:
        out_mask = masks_root / f"desi_lssmask_{tracer}{cap_tag}_nside{nside}.fits.gz"
        hp.write_map(out_mask, tracer_keep_union.astype(np.uint8), dtype=np.uint8,
                    nest=False, coord="C", overwrite=True, fits_IDL=False)


            

        LOGGER.info("Done tracer %s", tracer)

    # (Optionally) write a simple manifest for downstream steps
    manifest = maps_root / f"manifest_nside{nside}.json"
    with open(manifest, "w") as f:
        json.dump({"tracers": args.tracers, "nside": nside}, f, indent=2)
    LOGGER.info("Wrote manifest → %s", manifest)



if __name__ == "__main__":
    main()
