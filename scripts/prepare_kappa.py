#!/usr/bin/env python3
"""
Prepare the Planck CMB-lensing convergence map and the DESI∩lensing joint mask.

The Planck lensing product gives kappa as harmonic coefficients (kappa_lm, Lmax=4096),
not a pixel map. This script truncates to the target band, synthesises a kappa map at the
analysis NSIDE, and builds the joint mask = (Planck lensing mask) ∩ (DESI LSS footprint),
the analogue of the temperature joint mask but with the lensing analysis mask.

Outputs:
  maps/planck/nside{N}/kappa.fits.gz
  masks/planck_desi_kappa_joint_nside{N}.fits.gz
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import healpy as hp
import yaml

LOGGER = logging.getLogger("prepare_kappa")


def setup_logging(v):
    logging.basicConfig(level=logging.INFO if v else logging.WARNING,
                        format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--lmax", type=int, default=None, help="Band-limit for the kappa map (default 3*nside-1)")
    p.add_argument("--mask-thresh", type=float, default=0.5, help="Binarise the lensing mask above this")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def truncate_alm(alm, lmax_in, lmax_out):
    """Return alm restricted to L<=lmax_out (healpy m-major layout)."""
    out = np.zeros(hp.Alm.getsize(lmax_out), dtype=alm.dtype)
    for m in range(lmax_out + 1):
        n = lmax_out - m + 1
        i_in = hp.Alm.getidx(lmax_in, m, m)
        i_out = hp.Alm.getidx(lmax_out, m, m)
        out[i_out:i_out + n] = alm[i_in:i_in + n]
    return out


def load_mask_ring(path, nside, thresh=0.5, rotate_g2c=True):
    m, hdr = hp.read_map(path, dtype=np.float64, h=True)
    order = next((str(v).strip().upper() for k, v in hdr if k == "ORDERING"), "RING")
    if order == "NESTED":
        m = hp.reorder(m, n2r=True)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside, order_in="RING", order_out="RING")
    if rotate_g2c:  # Planck (Galactic) → Equatorial to match DESI maps
        m = hp.Rotator(coord=["G", "C"]).rotate_map_pixel(m)
    return (np.clip(m, 0, 1) > thresh).astype(np.float64)


def union_lss_masks(masks_dir: Path, nside: int):
    """Union of desi_lssmask_*_nside{N} (excluding cap-tagged) → DESI footprint."""
    npix = hp.nside2npix(nside)
    union = np.zeros(npix)
    pats = [p for p in masks_dir.glob(f"desi_lssmask_*_nside{nside}.fits.gz") if "_cap" not in p.name]
    for p in pats:
        m = hp.read_map(p, dtype=np.float64)
        if hp.get_nside(m) != nside:
            m = hp.ud_grade(m, nside_out=nside)
        union = np.maximum(union, (m > 0).astype(float))
    LOGGER.info("DESI LSS union from %d mask(s): fsky=%.3f", len(pats), np.mean(union > 0))
    return union


def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = yaml.safe_load(open(args.config))
    nside = args.nside
    lmax = args.lmax if args.lmax is not None else 3 * nside - 1

    pk = cfg["paths"]["planck"]
    klm_path = pk["kappa"]
    kmask_path = pk["kappa_mask"]

    # --- kappa map from kappa_lm ---
    klm = hp.read_alm(klm_path)
    lmax_in = hp.Alm.getlmax(klm.size)
    lmax = min(lmax, lmax_in)
    LOGGER.info("kappa_lm Lmax=%d → synthesising map at nside=%d, lmax=%d", lmax_in, nside, lmax)
    kmap = hp.alm2map(truncate_alm(klm, lmax_in, lmax), nside, lmax=lmax)
    # Planck products are in GALACTIC coords; DESI maps are pixelised in EQUATORIAL
    # (RA/Dec). Rotate G→C so the cross-correlation is in a single frame.
    kmap = hp.Rotator(coord=["G", "C"]).rotate_map_alms(kmap, lmax=lmax)
    LOGGER.info("kappa map (rotated G→C): mean=%.2e std=%.3e", kmap.mean(), kmap.std())

    maps_root = Path(cfg["paths"]["maps"]) / "planck" / f"nside{nside}"
    maps_root.mkdir(parents=True, exist_ok=True)
    out_kappa = maps_root / "kappa.fits.gz"
    hp.write_map(str(out_kappa), kmap.astype(np.float32), dtype=np.float32, overwrite=True, coord="G")
    LOGGER.info("Wrote kappa map → %s", out_kappa)

    # --- joint mask: lensing mask ∩ DESI footprint ---
    masks_root = Path(cfg["paths"]["masks"])
    lens_mask = load_mask_ring(kmask_path, nside, args.mask_thresh)
    LOGGER.info("Planck lensing mask: fsky=%.3f (at nside %d)", np.mean(lens_mask > 0), nside)
    lss_union = union_lss_masks(masks_root, nside)
    joint = lens_mask * lss_union
    out_mask = masks_root / f"planck_desi_kappa_joint_nside{nside}.fits.gz"
    hp.write_map(str(out_mask), joint.astype(np.float32), dtype=np.float32, overwrite=True, coord="G")
    LOGGER.info("Wrote DESI∩lensing joint mask → %s  (fsky=%.3f)", out_mask, np.mean(joint > 0))


if __name__ == "__main__":
    main()
