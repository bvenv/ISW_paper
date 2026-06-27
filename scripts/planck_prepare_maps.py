#!/usr/bin/env python3
"""
Prepare Planck 2018 CMB temperature maps for ISW cross-correlation.

For each requested label (SMICA primary; Commander/SEVEM optional):
  1) Load Planck T map (K_CMB), downgrade/upgrade to target NSIDE
  2) Build DESI∩Planck joint mask at target NSIDE
  3) Remove monopole & dipole on the JOINT MASK
  4) Optionally smooth to a target Gaussian FWHM (extra only; no deconvolution)
  5) Convert units (K_CMB → µK_CMB) if requested
  6) Apply mask (UNSEEN outside) and write:
       maps/planck/nside{NSIDE}/cmb_T_{label}.fits.gz
     Also writes the joint mask to:
       masks/planck_desi_joint_nside{NSIDE}.fits.gz
"""
import argparse
import logging
from pathlib import Path
import numpy as np
import healpy as hp
import yaml
import os

LOGGER = logging.getLogger("planck_prepare_maps")


# ---------- CLI / config ----------
def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="configs/desi.yml")
    p.add_argument("--nside", type=int, default=1024, help="Target NSIDE for output maps/mask")
    p.add_argument("--labels", nargs="+", default=["SMICA", "Commander", "SEVEM"],
                   help="Which Planck component maps to prepare (must have paths in YAML)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--units", choices=["K", "uK"], default="uK", help="Output units")
    p.add_argument("--fwhm-arcmin", type=float, default=0.0,
                   help="Optional Gaussian smoothing to apply AFTER mono/dipole removal (0 = none)")
    p.add_argument("--save-unmasked", action="store_true",
                   help="Also save a version without UNSEEN applied (rarely needed)")
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("--regress-y", action="store_true",
               help="Regress the Planck y-map (tSZ) from the CMB map on the joint mask")
    p.add_argument("--cmb-field", default="I_STOKES",
               help="Column to read from Planck IQU BINTABLE (e.g., I_STOKES or I_STOKES_INP)")
    p.add_argument("--planck-galmask-col", default="GAL090",
                help="Column name in the Galactic-plane mask (e.g., GAL090, GAL097, GAL099)")
    # Optional convenience if you prefer giving a fraction; ignored if --planck-galmask-col is set
    p.add_argument("--planck-galmask-frac", type=float, default=0.9,
                help="Desired sky fraction (e.g., 0.9, 0.97, 0.99). Chooses nearest available GALxxx column.")

    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- helpers ----------
from astropy.io import fits  # at top

def _read_planck_T(path, field_name="I_STOKES"):
    """
    Read a single column (default: I_STOKES) from a Planck IQU BINTABLE.
    Returns a float64 1D map in RING ordering.
    """
    with fits.open(path, memmap=True) as hdul:
        tab = hdul[1]
        hdr = tab.header
        arr = np.array(tab.data[field_name], dtype=np.float64)
        ordering = str(hdr.get("ORDERING", "RING")).strip().upper()
        nside = int(hdr.get("NSIDE", hp.npix2nside(arr.size)))
    m = arr
    if ordering == "NESTED":
        m = hp.reorder(m, n2r=True)
    if hp.get_nside(m) != nside:
        # Rely on header NSIDE for safety; no resample here (caller handles ud_grade)
        pass
    return m



def _to_target_nside(m, nside):
    if hp.get_nside(m) == nside:
        return m
    # Preserve units; ud_grade for T is fine at low-ell use cases
    return hp.ud_grade(m, nside_out=nside, order_in="RING", order_out="RING", power=None)


def _rotate_g2c_map(m, lmax=None):
    """Rotate a map from Galactic to Equatorial (C). Planck products are native Galactic;
    DESI maps are pixelised in Equatorial (RA/Dec), so the CMB must be rotated to cross-
    correlate in a single frame."""
    return hp.Rotator(coord=["G", "C"]).rotate_map_alms(m, lmax=lmax)


def _read_mask(path, nside, colname=None, frac=None):
    """
    Read a Planck Galactic mask column by name (e.g., 'GAL090').
    If frac is given (0<frac<1), picks nearest among GAL020,040,060,070,080,090,097,099.
    Returns a float mask in {0,1} at target NSIDE, RING ordering.
    """
    # Decide column
    choose = colname
    if choose is None and frac is not None:
        targets = [0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 0.97, 0.99]
        names   = ["GAL020","GAL040","GAL060","GAL070","GAL080","GAL090","GAL097","GAL099"]
        choose = names[int(np.argmin([abs(frac - t) for t in targets]))]

    with fits.open(path, memmap=True) as hdul:
        tab = hdul[1]
        hdr = tab.header
        # Default to first column if not specified
        if choose is None:
            choose = tab.columns.names[0]
        if choose not in tab.columns.names:
            raise ValueError(f"Mask column '{choose}' not found. Available: {tab.columns.names}")
        arr = np.array(tab.data[choose], dtype=np.float64)
        ordering = str(hdr.get("ORDERING", "RING")).strip().upper()
        src_nside = int(hdr.get("NSIDE", hp.npix2nside(arr.size)))
    m = arr
    if ordering == "NESTED":
        m = hp.reorder(m, n2r=True)
    # Some GAL masks are 0/1 bytes; ensure 0/1 float
    m = np.where(m > 0.5, 1.0, 0.0)
    if src_nside != nside:
        m = hp.ud_grade(m, nside_out=nside, order_in="RING", order_out="RING", power=None)
        m = np.where(m > 0.5, 1.0, 0.0)
    # Galactic → Equatorial to match the DESI footprint
    m = hp.Rotator(coord=["G", "C"]).rotate_map_pixel(m)
    m = np.where(m > 0.5, 1.0, 0.0)
    return m



def _union_lss_masks(masks_dir: Path, nside: int):
    """
    Union all DESI LSS masks of the form desi_lssmask_*_nside{nside}.fits.gz in masks_dir.
    Returns float mask in {0,1}.
    """
    pats = list(masks_dir.glob(f"desi_lssmask_*_nside{nside}.fits.gz"))
    if not pats:
        LOGGER.warning("No DESI LSS masks found at NSIDE=%d under %s", nside, masks_dir)
        return None
    acc = None
    for p in pats:
        m = hp.read_map(p, dtype=np.float64, verbose=False)
        m = np.where(m > 0.5, 1.0, 0.0)
        acc = m if acc is None else np.maximum(acc, m)
    return acc


def _build_joint_mask(cfg: dict, nside: int,args):
    masks_root = Path(cfg["paths"]["masks"])
    masks_root.mkdir(parents=True, exist_ok=True)

    # Planck union mask (can be aggressive; we'll AND it)
    p_union = cfg["paths"]["planck"].get("union_mask")
    if p_union and os.path.exists(p_union):
        try:
            planck_mask = _read_mask(
                p_union, nside,
                colname=args.planck_galmask_col,
                frac=args.planck_galmask_frac
            )
        except Exception as e:
            LOGGER.warning("Failed to read Planck union mask (%s); proceeding without it.", e)
            planck_mask = None
    else:
        LOGGER.warning("paths.planck.union_mask not set or missing; proceeding without it.")
        planck_mask = None


    # DESI union across tracers
    lss_union = _union_lss_masks(masks_root, nside)

    if lss_union is None and planck_mask is None:
        LOGGER.warning("No masks found; using all-sky mask (not ideal).")
        joint = np.ones(hp.nside2npix(nside), dtype=np.float64)
    elif lss_union is None:
        joint = planck_mask
    elif planck_mask is None:
        joint = lss_union
    else:
        joint = (lss_union * planck_mask)

    # Ensure strictly 0/1
    joint = np.where(joint > 0.5, 1.0, 0.0)

    out_mask = masks_root / f"planck_desi_joint_nside{nside}.fits.gz"
    hp.write_map(out_mask, joint.astype(np.uint8), dtype=np.uint8, nest=False, coord="C",
                 overwrite=True, fits_IDL=False)
    LOGGER.info("Wrote joint DESI∩Planck mask → %s", out_mask)
    return joint, out_mask


def _remove_monopole_dipole(m, mask):
    """
    Remove best-fit monopole and dipole using a weighted linear regression on the masked sky.
    m: full-sky map (float64), mask in {0,1}.
    Returns cleaned map.
    """
    good = (mask > 0.5) & np.isfinite(m)
    if good.sum() == 0:
        LOGGER.warning("Mask leaves no valid pixels for mono/dipole fit; returning original map.")
        return m

    ipix = np.where(good)[0]
    theta, phi = hp.pix2ang(hp.get_nside(m), ipix, nest=False)
    # Unit vectors
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    yvec = m[good]
    # Design matrix [1, x, y, z]
    X = np.vstack([np.ones_like(yvec), x, y, z]).T
    # Weighted least squares with equal weights (mask already applied)
    beta, *_ = np.linalg.lstsq(X, yvec, rcond=None)
    a, bx, by, bz = beta
    LOGGER.debug("Monopole (K_CMB): %.3e; Dipole (K_CMB): [%.3e, %.3e, %.3e]", a, bx, by, bz)

    # Subtract
    model = a + bx * x + by * y + bz * z
    m_clean = m.copy()
    m_clean[good] = yvec - model
    return m_clean


def _apply_mask_unseen(m, mask):
    out = np.full_like(m, hp.UNSEEN, dtype=np.float32)
    sel = mask > 0.5
    out[sel] = m[sel].astype(np.float32)
    return out


def _prepare_one(cfg, label: str, nside: int, outdir: Path, units: str, fwhm_arcmin: float,
                 joint_mask: np.ndarray, overwrite: bool, save_unmasked: bool,
                 regress_y: bool,cmb_field: str):
    # Map paths from config
    pmap = cfg["paths"]["planck"].get(label.lower()) or cfg["paths"]["planck"].get(label)
    if pmap is None:
        LOGGER.warning("No path for Planck label '%s' in YAML; skipping.", label)
        return
    if not os.path.exists(pmap):
        LOGGER.warning("Planck map not found: %s ; skipping %s.", pmap, label)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    out_map = outdir / f"cmb_T_{label}.fits.gz"
    out_unmasked = outdir / f"cmb_T_{label}_nomask.fits.gz"

    if out_map.exists() and not overwrite:
        LOGGER.info("Skip existing %s", out_map)
        return

    # 1) Read & NSIDE, then rotate Galactic → Equatorial (to match DESI maps)
    T =  _read_planck_T(pmap, field_name=cmb_field)   # K_CMB
    T = _to_target_nside(T, nside)
    T = _rotate_g2c_map(T, lmax=3 * nside - 1)

    # 2) Remove mono/dipole on the JOINT mask
    T_md = _remove_monopole_dipole(T, joint_mask)

    # 3) Optional smoothing (after mono/dipole removal)
    if fwhm_arcmin and fwhm_arcmin > 0:
        fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
        T_md = hp.smoothing(T_md, fwhm=fwhm_rad, verbose=False)

    # Optional: regress y-map (tSZ template) on the same mask
    y_beta = np.nan
    y_corr = np.nan
    if regress_y:
        y_path = cfg["paths"]["planck"].get("y_map")
        if y_path and os.path.exists(y_path):
            ymap = _read_y_map(y_path)
            ymap = _to_target_nside(ymap, nside)
            # if you smoothed T, match smoothing on the template too
            if fwhm_arcmin and fwhm_arcmin > 0:
                fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
                ymap = hp.smoothing(ymap, fwhm=fwhm_rad, verbose=False)
            T_md, y_beta, y_corr = _regress_template(T_md, ymap, joint_mask)
            LOGGER.info("%s: y-regression β=%.3e K/y, corr(T,y)=%+.3f (in-mask)", label, y_beta, y_corr)
        else:
            LOGGER.warning("regress-y set but paths.planck.y_map not found; skipping y-regression.")


    # 4) Units
    if units == "uK":
        T_md *= 1e6
        unit_str = "uK_CMB"
    else:
        unit_str = "K_CMB"

    # 5) QA metrics in-mask
    good = (joint_mask > 0.5) & np.isfinite(T_md)
    mean_in = float(np.mean(T_md[good])) if good.any() else float("nan")
    std_in = float(np.std(T_md[good])) if good.any() else float("nan")
    LOGGER.info("%s: mean=%+.3e %s, std=%.3e %s (in joint mask)", label, mean_in, unit_str, std_in, unit_str)

    # 6) Apply mask → UNSEEN
    T_out = _apply_mask_unseen(T_md, joint_mask)
    # 7) Write
    hdr = [
        ("NSIDE", nside),
        ("LABEL", label),
        ("UNITS", unit_str),
        ("FWHMARC", float(fwhm_arcmin)),
        ("MDREM", True, "Monopole/dipole removed on joint mask"),
        ("MEANIN", mean_in),
        ("STDIN", std_in),
        ("YREG", bool(regress_y)),
        ("YBETA", (float(y_beta) if bool(regress_y)==True else False)),
        ("YCORR", (float(y_corr) if bool(regress_y)==True else False)),

    ]
    # print('[INFO] writing map ${out_map} with hdr ', hdr)
    hp.write_map(out_map, T_out, dtype=np.float32, nest=False, coord="C",
                 overwrite=True, fits_IDL=False, extra_header=hdr)
    LOGGER.info("Wrote cleaned CMB map → %s", out_map)

    if save_unmasked:
        hp.write_map(out_unmasked, T_md.astype(np.float32), dtype=np.float32, nest=False, coord="C",
                     overwrite=True, fits_IDL=False, extra_header=hdr)
        LOGGER.info("Wrote unmasked CMB map → %s", out_unmasked)

def _read_y_map(path):
    # Planck y-map is usually a single-temperature-like field (dimensionless y)
    y = hp.read_map(path, field=0, dtype=np.float64, verbose=False)
    return y

def _regress_template(T, template, mask):
    """
    Regress 'template' (e.g., y-map) out of T on the given binary mask.
    Returns (T_clean, beta, corr), where beta is the LS coefficient and
    corr is the Pearson correlation within the mask (for QA).
    """
    good = (mask > 0.5) & np.isfinite(T) & np.isfinite(template)
    if good.sum() == 0:
        return T, np.nan, np.nan
    t = template[good]
    y = T[good]
    denom = np.dot(t, t)
    if denom <= 0 or not np.isfinite(denom):
        return T, np.nan, np.nan
    beta = float(np.dot(t, y) / denom)
    T_clean = T.copy()
    T_clean[good] = y - beta * t
    # correlation (for log/debug)
    yt = y - y.mean()
    tt = t - t.mean()
    corr = float(np.dot(yt, tt) / (np.sqrt(np.dot(yt, yt)) * np.sqrt(np.dot(tt, tt)) + 1e-30))
    return T_clean, beta, corr



# ---------- main ----------
def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)

    maps_root = Path(cfg["paths"]["maps"]) / "planck" / f"nside{args.nside}"
    maps_root.mkdir(parents=True, exist_ok=True)

    # Build joint DESI∩Planck mask at this NSIDE (and write it)
    joint_mask, joint_path = _build_joint_mask(cfg, args.nside,args=args)

    for label in args.labels:
        _prepare_one(cfg, label=label, nside=args.nside, outdir=maps_root,
                    units=args.units, fwhm_arcmin=args.fwhm_arcmin,
                    joint_mask=joint_mask, overwrite=args.overwrite,
                    save_unmasked=args.save_unmasked, regress_y=args.regress_y, cmb_field=args.cmb_field)



if __name__ == "__main__":
    main()
