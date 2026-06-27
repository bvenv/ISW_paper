#!/usr/bin/env python3
"""
build_isw_templates_clank.py

Build *per-tomographic-bin* theory templates for DESI×Planck ISW:
  - C_ell^{gT,fid}(z_ibin)  ISW×galaxy cross   -> gT_{TRACER}_z{ibin}.npz   (µK, raw C_ell)
  - C_ell^{gg,fid}(z_ibin)  galaxy auto         -> gg_{TRACER}_z{ibin}.npz   (dimensionless, incl. b²)
  - C_ell^{TT} from CAMB                        -> cltt_camb_uK2.txt          (µK², raw C_ell)

Theory engine: **CAMB source windows** (full Boltzmann, NOT Limber). The galaxy `counts`
window (this bin's n(z) + linear bias) is cross-correlated with the CMB temperature; TxW1 is
the ISW×galaxy cross and W1xW1 the galaxy auto. This is essential for the ISW: it lives at
low ℓ where Limber fails, and clank-v4's ISWTracer kernel is currently wrong (wrong sign,
~10× amplitude, shape error — see scripts/validate_isw_template.py). clank's *gg* is fine,
but we take gg from the same CAMB call so gT and gg share identical n(z)/bias/cosmology.

(The filename is kept for the orchestrator/PDF; "clank" is historical.)

Per-bin n(z) come from make_perbin_nz.py; per-bin bias from the calibrated priors when
available, else a power-law fallback b(z)=b0*(1+z)^alpha.
"""
from pathlib import Path
import argparse
import json
import logging

import numpy as np
import yaml
import camb
from camb.sources import SplinedSourceWindow

LOGGER = logging.getLogger("build_isw_templates")

# Power-law bias fallback (b0, alpha) per tracer when no calibrated prior is present.
BIAS_FALLBACK = {"BGS": (1.3, 0.5), "LRG": (2.0, 0.6), "ELG": (1.4, 0.8), "QSO": (2.3, 1.0)}

# CAMB counts source terms to disable, leaving pure galaxy density × ISW.
_COUNTS_OFF = ("counts_redshift", "counts_lensing", "counts_velocity", "counts_radial",
               "counts_timedelay", "counts_ISW", "counts_potential", "counts_evolve")


def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config (bins + theory.cosmo)")
    ap.add_argument("--nz-dir", default="templates/isw/nz",
                    help="Directory with per-bin n(z): {TRACER}_z{ibin}_nz.txt")
    ap.add_argument("--outdir", default="templates/isw")
    ap.add_argument("--lmax", type=int, default=200)
    ap.add_argument("--tracers", nargs="+", default=None)
    ap.add_argument("--bias-priors", default="results/desi_bias_priors.json")
    ap.add_argument("--skip-tt", action="store_true")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    return ap.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cosmo_params(cfg):
    c = cfg.get("theory", {}).get("cosmo", {})
    return dict(H0=float(c.get("H0", 67.36)), ombh2=float(c.get("ombh2", 0.02237)),
                omch2=float(c.get("omch2", 0.1200)), ns=float(c.get("ns", 0.9649)),
                As=float(c.get("As", 2.1e-9)), tau=float(c.get("tau", 0.054)))


def read_nz(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    z = arr[:, 0]
    nz = arr[:, 3] if arr.shape[1] >= 4 else arr[:, 1]
    nz = np.clip(nz, 0.0, None)
    return z, nz / np.trapz(nz, z)


def load_bias_priors(path):
    p = Path(path)
    if not p.exists():
        LOGGER.info("No bias priors at %s; using power-law fallback", path)
        return {}
    data = json.load(open(p))
    return {(e["tracer"], int(e["ibin"])): float(e["b"])
            for e in data.get("priors", []) if float(e.get("b", 0)) > 0}


def camb_window_cls(cp, z, nz, bias, lmax):
    """Return (ell, gT[µK], gg[dimensionless]) from CAMB source windows (density×ISW)."""
    pars = camb.set_params(H0=cp["H0"], ombh2=cp["ombh2"], omch2=cp["omch2"],
                           ns=cp["ns"], As=cp["As"], tau=cp["tau"])
    pars.SourceWindows = [SplinedSourceWindow(z=z, W=nz, source_type="counts", bias=bias)]
    pars.SourceTerms.counts_density = True
    for term in _COUNTS_OFF:
        setattr(pars.SourceTerms, term, False)
    pars.SourceTerms.limber_windows = False
    pars.set_for_lmax(lmax + 50)
    res = camb.get_results(pars)
    d = res.get_cmb_unlensed_scalar_array_dict(CMB_unit="muK", raw_cl=True)
    gT, gg = d["TxW1"][: lmax + 1], d["W1xW1"][: lmax + 1]
    ell = np.arange(gT.size)
    return ell, gT, gg


def write_camb_tt(cp, lmax, out_path):
    pars = camb.set_params(H0=cp["H0"], ombh2=cp["ombh2"], omch2=cp["omch2"],
                           ns=cp["ns"], As=cp["As"], tau=cp["tau"])
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    powers = camb.get_results(pars).get_cmb_power_spectra(pars, lmax=lmax, raw_cl=True,
                                                          CMB_unit="muK")
    cltt = powers["total"][:, 0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.c_[np.arange(len(cltt)), cltt], fmt=["%d", "%.10e"])
    LOGGER.info("Wrote CAMB TT -> %s", out_path)


def save_npz(path, ell, cl):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ell=np.asarray(ell, int), cl=np.asarray(cl, float))


def main():
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    cp = cosmo_params(cfg)
    bins_cfg = cfg["desi"]["bins"]
    tracers = args.tracers or list(bins_cfg.keys())
    outdir, nz_dir = Path(args.outdir), Path(args.nz_dir)
    priors = load_bias_priors(args.bias_priors)

    for tracer in tracers:
        b0, alpha = BIAS_FALLBACK.get(tracer, (2.0, 0.6))
        for idx, (zmin, zmax) in enumerate(bins_cfg[tracer]):
            ibin = idx + 1
            nz_path = nz_dir / f"{tracer}_z{ibin}_nz.txt"
            if not nz_path.exists():
                LOGGER.warning("Missing per-bin n(z) %s — skipping %s z%d", nz_path, tracer, ibin)
                continue
            z, nz = read_nz(nz_path)
            b = priors.get((tracer, ibin))
            if b is not None:
                btag = f"b={b:.3f} (prior)"
            else:
                b = b0 * (1.0 + 0.5 * (zmin + zmax)) ** alpha
                btag = f"b={b:.3f} (power-law)"

            ell, gT, gg = camb_window_cls(cp, z, nz, b, args.lmax)
            save_npz(outdir / f"gT_{tracer}_z{ibin}.npz", ell, gT)
            save_npz(outdir / f"gg_{tracer}_z{ibin}.npz", ell, gg)
            LOGGER.info("%s z%d [%.2f,%.2f) %s -> gT/gg (CAMB TxW1/W1xW1)",
                        tracer, ibin, zmin, zmax, btag)

    if not args.skip_tt:
        write_camb_tt(cp, args.lmax, outdir / "cltt_camb_uK2.txt")
    LOGGER.info("Done. Per-bin templates in %s", outdir)


if __name__ == "__main__":
    main()
