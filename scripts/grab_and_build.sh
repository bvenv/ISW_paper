#!/usr/bin/env bash
# grab_and_build.sh — download DESI DR1 LSScats for a tracer, build maps + per-bin n(z).
# Processes tracers serially. By default KEEPS the catalogs on disk (KEEP=1); set KEEP=0 to
# delete them after each build (then peak disk ~ one tracer; built maps are only ~20 MB each).
#
#   bash scripts/grab_and_build.sh [BGS ELG QSO]        # keep catalogs (default)
#   KEEP=0 bash scripts/grab_and_build.sh BGS ELG QSO   # delete catalogs after each build
#
# Notes:
#  * 2 random chunks per cap (ample for the nside-512 selection function; halves the download).
#  * After all tracers, rebuilds the DESI∩Planck joint masks (T + kappa) so each tracer's
#    cross-correlation uses the correct, updated footprint (downstream-correct).
set -euo pipefail

PY=/home/brandon/anaconda3/envs/clankv4_dev/bin/python
REPO=/home/brandon/PHD/ISW
CFG=$REPO/configs/desi.yml
BASE="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/"
DATA=/home/brandon/PHD/Data/DESI
NCHUNK=2          # random chunks per cap
NSIDE=512

declare -A PREFIX=( [BGS]=BGS_BRIGHT [ELG]=ELG_LOPnotqso [QSO]=QSO [LRG]=LRG )
TRACERS=( "${@:-BGS ELG QSO}" )
TRACERS=( ${TRACERS[@]} )

cd "$REPO"
echo "=== grab_and_build: ${TRACERS[*]}  (nside=$NSIDE, $NCHUNK random chunks/cap) ==="
df -h /home/brandon | tail -1

for T in "${TRACERS[@]}"; do
  pre=${PREFIX[$T]:-$T}
  OUT=$DATA/$T
  mkdir -p "$OUT"
  echo; echo "######## $T  (prefix $pre) ########"
  free=$(df --output=avail -BG /home/brandon | tail -1 | tr -dc '0-9')
  echo "  free: ${free} GB"
  if [ "$free" -lt 12 ]; then echo "  ! <12 GB free — aborting"; exit 1; fi

  echo "  [1/4] downloading data + $NCHUNK random chunks/cap ..."
  for cap in NGC SGC; do
    wget -qc -P "$OUT" "${BASE}${pre}_${cap}_clustering.dat.fits"
    for i in $(seq 0 $((NCHUNK-1))); do
      wget -qc -P "$OUT" "${BASE}${pre}_${cap}_${i}_clustering.ran.fits"
    done
  done
  echo "  downloaded: $(du -sh "$OUT" | cut -f1)"

  echo "  [2/4] building delta maps (cap=ANY) ..."
  $PY scripts/desi_build_maps.py --config "$CFG" --tracers "$T" --nside $NSIDE --cap ANY --overwrite \
      2>&1 | grep -E 'Done tracer|objects|FileNotFound|Error' | tail -4

  echo "  [3/4] building per-bin n(z) (catalog-weighted) ..."
  $PY scripts/make_perbin_nz.py --config "$CFG" --tracers "$T" 2>&1 | grep -E 'wrote|Tracer' | tail -4

  if [ "${KEEP:-1}" = "1" ]; then
    echo "  [4/4] KEEP=1 — keeping catalogs on disk (re-usable for per-cap maps, nside changes)"
  else
    echo "  [4/4] deleting catalogs (keeping ~20 MB of maps) ..."
    rm -f "$OUT"/*.fits
  fi
  echo "  done $T.  free now: $(df --output=avail -BG /home/brandon | tail -1 | tr -d ' ')"
done

echo; echo "=== rebuilding DESI∩Planck joint masks for the updated footprint ==="
$PY scripts/planck_prepare_maps.py --config "$CFG" --nside $NSIDE --overwrite 2>&1 | grep -iE 'joint|wrote cleaned' | head
$PY scripts/prepare_kappa.py --config "$CFG" --nside $NSIDE 2>&1 | grep -iE 'joint mask' | tail -1
echo "=== grab_and_build complete for: ${TRACERS[*]} ==="
