#!/usr/bin/env python3
"""
Validate the ISW × galaxy cross-spectrum (gT) template against CAMB.

Reference: CAMB source windows (full Boltzmann, NOT Limber — essential because ISW lives
at low ℓ where Limber fails). A `counts` window with the bin n(z) and linear bias is
cross-correlated with the CMB temperature (`TxW1`), isolating the galaxy-density × ISW term
(RSD/lensing/velocity terms switched off). CAMB TxW1 is in µK (raw C_ell), matching the
measured δ_g × T(µK) spectra.

Compares CAMB TxW1 to:
  - the clank gT template (templates/isw/gT_{TRACER}_z{ibin}.npz), and/or
  - a CAMB-built gT template, if you switch the template builder to CAMB.

A correct ISW×galaxy cross is POSITIVE at low ℓ and falls with ℓ. As of this writing the
clank ISWTracer template is wrong (negative sign, ~10× too large, ℓ-dependent shape error),
so the gT template should be built from CAMB TxW1 until the clank ISW kernel is fixed.
"""
import argparse
import numpy as np
import camb
from camb.sources import SplinedSourceWindow


def camb_txw_reference(nz_path, bias, h, ombh2, omch2, ns, As, lmax):
    pars = camb.set_params(H0=100 * h, ombh2=ombh2, omch2=omch2, ns=ns, As=As, tau=0.054)
    z, nz = np.loadtxt(nz_path, unpack=True)
    nz = nz / np.trapz(nz, z)
    pars.SourceWindows = [SplinedSourceWindow(z=z, W=nz, source_type="counts", bias=bias)]
    st = pars.SourceTerms
    st.counts_density = True
    for term in ("counts_redshift", "counts_lensing", "counts_velocity", "counts_radial",
                 "counts_timedelay", "counts_ISW", "counts_potential", "counts_evolve"):
        setattr(st, term, False)
    st.limber_windows = False
    pars.set_for_lmax(lmax + 50)
    res = camb.get_results(pars)
    d = res.get_cmb_unlensed_scalar_array_dict(CMB_unit="muK", raw_cl=True)
    return d["TxW1"]  # µK, raw C_ell, indexed by ell


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--nz", default="templates/isw/nz/LRG_z1_nz.txt")
    ap.add_argument("--template", default="templates/isw/gT_LRG_z1.npz",
                    help="clank/CAMB gT template NPZ to compare against (ell, cl)")
    ap.add_argument("--bias", type=float, default=1.905)
    ap.add_argument("--ells", type=int, nargs="+", default=[6, 10, 20, 40, 80])
    ap.add_argument("--h", type=float, default=0.6736)
    ap.add_argument("--ombh2", type=float, default=0.02237)
    ap.add_argument("--omch2", type=float, default=0.1200)
    ap.add_argument("--ns", type=float, default=0.9649)
    ap.add_argument("--As", type=float, default=2.1e-9)
    args = ap.parse_args()

    lmax = max(args.ells)
    ref = camb_txw_reference(args.nz, args.bias, args.h, args.ombh2, args.omch2,
                             args.ns, args.As, lmax)
    tmpl = np.load(args.template)

    print(f"n(z): {args.nz}   bias={args.bias}   template={args.template}")
    print(" ell    CAMB TxW1[µK]    template        template/CAMB")
    for l in args.ells:
        j = int(np.where(tmpl["ell"] == l)[0][0])
        r = tmpl["cl"][j] / ref[l] if ref[l] != 0 else np.nan
        print(f"{l:4d}    {ref[l]: .4e}    {tmpl['cl'][j]: .4e}    {r:8.3f}")
    print("\nA correct gT template should match CAMB TxW1 (ratio ~ +1, same sign).")


if __name__ == "__main__":
    main()
