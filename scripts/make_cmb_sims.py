#!/usr/bin/env python3
"""
Generate Gaussian CMB T simulations for empirical bandpower covariance estimation.

Draws nsims independent realisations from C_ell^{TT}, applies the Planck×DESI
joint mask, and writes indexed FITS maps.  These can then be cross-correlated
with DESI delta maps to build an empirical bandpower covariance that validates
(or replaces) the Gaussian approximation used in fit_isw_amplitudes.py.

Workflow
--------
  1. Load C_ell^{TT} from templates/isw/cltt_camb_uK2.txt  (or --cltt)
  2. Load the Planck×DESI joint mask at the target NSIDE
  3. For each sim i in [0, nsims):
       a. Draw alm from C_ell^{TT} with hp.synalm (independent seed per sim)
       b. Apply Gaussian beam transfer function (optional, --beam-fwhm-arcmin)
       c. Apply HEALPix pixel window
       d. Project to map: hp.alm2map
       e. Set pixels outside mask to hp.UNSEEN
       f. Write sims/cmb/sim_{i:04d}.fits.gz
  4. Write metadata JSON for reproducibility

Inputs
------
  templates/isw/cltt_camb_uK2.txt               (or --cltt)
  masks/planck_desi_joint_nside{N}.fits.gz       (or --mask-path)

Outputs
-------
  {outdir}/sim_{i:04d}.fits.gz      i = 0 .. nsims-1
  {outdir}/meta.json
"""
import argparse
import json
import logging
from pathlib import Path

import healpy as hp
import numpy as np
import yaml

LOGGER = logging.getLogger("make_cmb_sims")


# ---------- CLI ----------

def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--nsims", type=int, default=200)
    p.add_argument("--lmax", type=int, default=None,
                   help="Max ell for alm synthesis (default: 3*nside-1)")
    p.add_argument("--beam-fwhm-arcmin", type=float, default=0.0,
                   help="FWHM of Gaussian beam to apply in addition to pixel window "
                        "(0 = pixel window only).  Match to the real CMB map beam.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed; sim i uses SeedSequence([seed, i]) for independence")
    p.add_argument("--cltt", type=str, default=None,
                   help="C_ell^TT file (2-col text: ell clTT µK²). "
                        "Default: templates/isw/cltt_camb_uK2.txt")
    p.add_argument("--mask-path", type=str, default=None,
                   help="Joint mask FITS path. Default: read from config paths.masks")
    p.add_argument("--outdir", default="sims/cmb/")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------- helpers ----------

def load_cltt(path: Path, lmax: int) -> np.ndarray:
    """Load two-column C_ell^TT file and return array indexed 0..lmax."""
    arr = np.loadtxt(path)
    ell_in = arr[:, 0].astype(int)
    cl_in  = arr[:, 1].astype(float)
    cl = np.zeros(lmax + 1, dtype=float)
    valid = (ell_in >= 0) & (ell_in <= lmax)
    cl[ell_in[valid]] = cl_in[valid]
    return cl


def gaussian_beam(fwhm_arcmin: float, lmax: int) -> np.ndarray:
    """Return Gaussian beam transfer function b_ell (length lmax+1)."""
    if fwhm_arcmin <= 0.0:
        return np.ones(lmax + 1)
    fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1, dtype=float)
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma**2)


def load_mask(path: str, nside: int) -> np.ndarray:
    """Load a HEALPix mask into RING ordering at target nside, clipped to [0,1]."""
    m, hdr = hp.read_map(path, dtype=np.float64, h=True, verbose=False)
    ordering = "RING"
    for k, v in hdr:
        if k == "ORDERING":
            ordering = str(v).strip().upper()
            break
    if ordering == "NESTED":
        m = hp.reorder(m, n2r=True)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside, order_in="RING", order_out="RING", power=None)
    return np.clip(m, 0.0, 1.0)


# ---------- main ----------

def main():
    args = parse_args()
    setup_logging(args.verbose)

    cfg = load_config(args.config)
    lmax   = args.lmax if args.lmax is not None else 3 * args.nside - 1
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- C_ell^TT ---
    cltt_path = Path(args.cltt) if args.cltt else Path("templates/isw/cltt_camb_uK2.txt")
    if not cltt_path.exists():
        raise FileNotFoundError(
            f"C_ell^TT not found: {cltt_path}\n"
            "Run build_isw_templates_clank.py first, or pass --cltt.")
    cltt = load_cltt(cltt_path, lmax)
    LOGGER.info("C_ell^TT: %s  (lmax=%d, peak=%.3g µK² at ell=%d)",
                cltt_path, lmax, cltt.max(), int(cltt.argmax()))

    # --- Mask ---
    if args.mask_path:
        mask_path = Path(args.mask_path)
    else:
        masks_root = Path(cfg["paths"]["masks"])
        mask_path = masks_root / f"planck_desi_joint_nside{args.nside}.fits.gz"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Joint mask not found: {mask_path}\n"
            "Run planck_prepare_maps.py first, or pass --mask-path.")
    mask = load_mask(str(mask_path), args.nside)
    fsky = float(np.mean(mask > 0))
    LOGGER.info("Mask: %s  (f_sky=%.4f)", mask_path, fsky)

    # --- Transfer function: beam × pixel window ---
    bl = gaussian_beam(args.beam_fwhm_arcmin, lmax)
    pw = hp.pixwin(args.nside, lmax=lmax)
    # pixwin returns (TT, PP) for pol; for temperature take first element if needed
    if pw.ndim > 1:
        pw = pw[0]
    pw = pw[:lmax + 1]
    bl_total = bl * pw

    # --- Optional progress bar ---
    try:
        from tqdm import tqdm
        itr = tqdm(range(args.nsims), desc="CMB sims", unit="sim")
    except ImportError:
        itr = range(args.nsims)

    # --- Generate sims ---
    n_written = n_skipped = 0
    for i in itr:
        out_path = outdir / f"sim_{i:04d}.fits.gz"
        if out_path.exists() and not args.overwrite:
            LOGGER.debug("Skip existing %s", out_path.name)
            n_skipped += 1
            continue

        # Derive a unique, reproducible 32-bit seed for each sim
        seed_i = int(np.random.SeedSequence([args.seed, i]).generate_state(1)[0])
        np.random.seed(seed_i)
        alm = hp.synalm(cltt, lmax=lmax, new=True)

        # Apply transfer function (beam + pixel window)
        hp.almxfl(alm, bl_total, inplace=True)

        T_map = hp.alm2map(alm, args.nside, lmax=lmax, verbose=False)

        # Mask: pixels outside footprint set to UNSEEN
        T_map[mask <= 0] = hp.UNSEEN

        hp.write_map(str(out_path), T_map, dtype=np.float32,
                     overwrite=True, coord="G")
        n_written += 1
        LOGGER.debug("Wrote sim %04d  (seed_i=%d)", i, seed_i)

    LOGGER.info("Wrote %d new sims, skipped %d existing", n_written, n_skipped)

    # --- Metadata ---
    meta = {
        "seed": args.seed,
        "nsims": args.nsims,
        "nside": args.nside,
        "lmax": lmax,
        "beam_fwhm_arcmin": args.beam_fwhm_arcmin,
        "pixel_window_applied": True,
        "cltt_path": str(cltt_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "fsky": round(fsky, 6),
        "n_written": n_written,
        "n_skipped": n_skipped,
    }
    meta_path = outdir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done: {n_written} sims written to {outdir}/")
    print(f"  metadata: {meta_path}")


if __name__ == "__main__":
    main()
