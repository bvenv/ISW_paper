#!/usr/bin/env bash
# run_xcorr_all.sh — full DESI×Planck cross-correlation chain for every tracer, then the
# multi-tracer A_ISW(z) combine. Assumes maps + n(z) exist (built by grab_and_build.sh) and
# the joint masks are current. Per tracer:
#   gT spectra -> kg spectra -> fit_bias_growth (kappa-pinned b) -> gT templates
#   -> fit_isw_amplitudes -> sim_covariance ;  then combine_aisw over all tracers.
set -euo pipefail
PY=/home/brandon/anaconda3/envs/clankv4_dev/bin/python
CFG=configs/desi.yml
SPEC=/home/brandon/PHD/ISWresults/spectra/desi
TRACERS=( "${@:-LRG BGS ELG QSO}" ); TRACERS=( ${TRACERS[@]} )
mkdir -p results/bias results/aisw
cd /home/brandon/PHD/ISW

for T in "${TRACERS[@]}"; do
  echo; echo "################  $T  ################"
  echo "[1] gT spectra";  $PY scripts/compute_crosscls.py --config $CFG --tracers $T --nside 512 --ell-max 150 --cmb-field T      --overwrite 2>&1 | grep -E 'f_sky\(apod|Wrote spectrum' | tail -3
  echo "[2] kg spectra";  $PY scripts/compute_crosscls.py --config $CFG --tracers $T --nside 512 --ell-max 150 --cmb-field kappa  --overwrite 2>&1 | grep -E 'Wrote spectrum' | tail -3
  echo "[3] bias (gg+kg)"; $PY scripts/fit_bias_growth.py --config $CFG --tracer $T --nside 512 --ell-max 150 --out results/bias/${T}_bias_kappa.json 2>&1 | grep -E 'z[0-9] ' | tail -3
  echo "[4] templates";   $PY scripts/build_isw_templates_clank.py --config $CFG --tracers $T --lmax 150 --bias-priors results/bias/${T}_bias_kappa.json 2>&1 | grep -E 'Done' | tail -1
  echo "[5] fit A_ISW";   $PY scripts/fit_isw_amplitudes.py --config $CFG --spectra-glob "$SPEC/gT_${T}_*_lmax150.npz" --templates-dir templates/isw --bias-priors results/bias/${T}_bias_kappa.json --outfile results/aisw/${T}_Aisw_table.csv 2>&1 | grep -E 'GLS'
done

echo; echo "################  JOINT covariance (all tracers, one CMB) + combine  ################"
$PY scripts/sim_covariance_joint.py --config $CFG --tracers ${TRACERS[*]} --nside 512 --ell-max 150 --nsims 300 2>&1 | grep -E 'Joint cov|Wrote|sim [0-9]+0/'
$PY scripts/combine_joint_aisw.py --config $CFG ${DROP:+--drop $DROP} 2>&1 | grep -E 'naive A|joint A|joint:'
echo "=== run_xcorr_all complete ==="
