#!/usr/bin/env bash
#SBATCH --job-name=dl_lss
#SBATCH --output=/scratch/pawsey0272/bvenvillle/isw/logs/download_desi_lss_dr1_%A_%a.out
#SBATCH --error=/scratch/pawsey0272/bvenvillle/isw/logs/download_desi_lss_dr1_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --partition=copy
#SBATCH --account=pawsey0272



# grab_desi_dr1_ls.sh — parallel download of DESI DR1 LSScats v1.5 for ISW work
# What it fetches:
#   * *_clustering.dat.fits      (data you pixelize)
#   * *_clustering.ran.fits      (randoms for selection function; many chunks)
#   * *_full.dat.fits            (full HPmapcut catalogs for n(z)/validation)
#   * *_nz.txt, *_frac_tlobs.fits (useful aux)
# What it skips: *_full.ran.fits, noveto variants, HPmapcut randoms (enormous).

set -euo pipefail

BASE="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/"
OUT="/scratch/pawsey0272/bvenvillle/isw/desi_dr1_ls/LSScats/v1.5"
mkdir -p "$OUT"

echo "[1/3] Building URL list from index…"
curl -fsSL "$BASE" \
| grep -oP '(?<=href=")[^"]+' \
| grep -E '^(BGS|LRG|ELG|QSO).*' \
| grep -E '(clustering\.dat\.fits|clustering\.ran\.fits|_full\.dat\.fits|_nz\.txt|_frac_tlobs\.fits)$' \
| grep -Ev '(_full\.ran\.fits|noveto|HPmapcut.*ran\.fits)' \
| sed "s#^#$BASE#" > "$OUT/urls.txt"

# (Optional) also grab pre-binned HEALPix maps from the hpmaps/ subdir
# Uncomment to include:
curl -fsSL "${BASE}hpmaps/" \
| grep -oP '(?<=href=")[^"]+' \
| grep -E '\.fits(\.gz)?$' \
| sed "s#^#${BASE}hpmaps/#" >> "$OUT/urls.txt"

NL=$(wc -l < "$OUT/urls.txt" || echo 0)
echo "   → Collected $NL file URLs into $OUT/urls.txt"

echo "[2/3] Starting parallel download…"
if command -v aria2c >/dev/null 2>&1; then
  echo "   Using aria2c (multi-connection, multi-file)."
  # -x: connections/server, -s: splits per file, -j: parallel files, -c: continue
  aria2c -i "$OUT/urls.txt" -d "$OUT" -c -x16 -s16 -j8 --summary-interval=5
else
  echo "   aria2c not found — falling back to wget + xargs parallelism."
  # Adjust -P (parallelism) to taste; 8–16 is usually safe on HPC
  cat "$OUT/urls.txt" | xargs -n1 -P24 -I{} wget -c -P "$OUT" {}
fi

echo "[3/3] Done. Files are in $OUT"
