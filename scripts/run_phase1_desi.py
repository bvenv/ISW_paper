#!/usr/bin/env python3
"""
Per-tracer orchestrator for the DESI×Planck tomographic ISW pipeline.

Runs the full chain for one or more tracers, each tracer independently, writing
per-tracer bias priors and A_ISW tables so tracers never clobber each other:

  build_maps -> make_perbin_nz -> compute_crosscls -> calibrate_bias
             -> build_isw_templates_clank -> fit_isw_amplitudes

Then run scripts/combine_aisw.py to assemble the A_ISW(z) curve across tracers.

Same command works locally (nside 512, LRG) and on Pawsey (nside 1024, all tracers);
only --nside / --tracers / paths-in-config differ. Run under the clankv4_dev env, e.g.
  conda run -n clankv4_dev python scripts/run_phase1_desi.py --config configs/desi.yml --tracers LRG
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

LOGGER = logging.getLogger("run_phase1_desi")

# Pipeline steps in order. Steps that read catalogs (maps, nz) are skipped
# automatically if the catalogs are absent but products already exist.
STEP_ORDER = ["maps", "nz", "xcls", "bias", "templates", "fit"]


def setup_logging(v):
    level = logging.WARNING if v == 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tracers", nargs="+", default=None,
                   help="Tracers to run (default: all in cfg['desi']['bins'])")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--skip", nargs="*", default=[],
                   help=f"Steps to skip: {', '.join(STEP_ORDER)}")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these steps (subset of the chain)")
    p.add_argument("--qa", action="store_true", help="Write map QA plots")
    p.add_argument("--dry-run", action="store_true", help="Print commands, do not run")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def run(cmd, dry):
    LOGGER.info("$ %s", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    cfg = yaml.safe_load(open(args.config))
    tracers = args.tracers or list(cfg["desi"]["bins"].keys())
    nside, L = args.nside, args.ell_max
    py = sys.executable
    here = Path(__file__).resolve().parent
    spectra_root = Path(cfg["paths"]["spectra"]) / "desi"

    bias_dir = Path("results/bias"); bias_dir.mkdir(parents=True, exist_ok=True)
    aisw_dir = Path("results/aisw"); aisw_dir.mkdir(parents=True, exist_ok=True)

    steps = set(STEP_ORDER) if args.only is None else set(args.only)
    steps -= set(args.skip)

    for tracer in tracers:
        LOGGER.info("==== tracer %s (nside=%d, lmax=%d) ====", tracer, nside, L)
        bias_priors = bias_dir / f"{tracer}_bias_priors.json"
        aisw_table = aisw_dir / f"{tracer}_Aisw_table.csv"

        cmds = {
            "maps": [py, str(here / "desi_build_maps.py"), "--config", args.config,
                     "--tracers", tracer, "--nside", str(nside), "--cap", "ANY", "--overwrite"]
                    + (["--qa"] if args.qa else []),
            "nz": [py, str(here / "make_perbin_nz.py"), "--config", args.config,
                   "--tracers", tracer],
            "xcls": [py, str(here / "compute_crosscls.py"), "--config", args.config,
                     "--tracers", tracer, "--nside", str(nside), "--ell-max", str(L), "--overwrite"],
            "bias": [py, str(here / "calibrate_bias.py"), "--config", args.config,
                     "--tracers", tracer, "--nside", str(nside), "--ell-max", str(L),
                     "--nz-dir", "templates/isw/nz", "--outfile", str(bias_priors)],
            "templates": [py, str(here / "build_isw_templates_clank.py"), "--config", args.config,
                          "--tracers", tracer, "--lmax", str(L), "--bias-priors", str(bias_priors)],
            "fit": [py, str(here / "fit_isw_amplitudes.py"), "--config", args.config,
                    "--spectra-glob", str(spectra_root / f"gT_{tracer}_*_lmax{L}.npz"),
                    "--templates-dir", "templates/isw", "--bias-priors", str(bias_priors),
                    "--outfile", str(aisw_table)],
        }

        for name in STEP_ORDER:
            if name not in steps:
                LOGGER.info("  -- skip %s", name)
                continue
            run(cmds[name], args.dry_run)

    LOGGER.info("Done. Next: python scripts/combine_aisw.py --config %s --tables 'results/aisw/*.csv'",
                args.config)


if __name__ == "__main__":
    main()
