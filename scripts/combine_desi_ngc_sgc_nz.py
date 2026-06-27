#!/usr/bin/env python3
"""
Combine DESI DR1 N(z) tables for NGC + SGC, and make an n(z) plot.

Expected input format (per line after comments):
  zmid  zlow  zhigh  n(z)  Nbin  Vol_bin

We combine by summing counts and volumes:
  N_tot = N_NGC + N_SGC
  V_tot = V_NGC + V_SGC
  n(z)  = N_tot / V_tot

We also (optionally) compute dN/dz per steradian and a normalized p(z) from N_tot and
the "effective area" in the file header.

Examples
--------
# simplest (uses default filenames in the current directory):
python combine_desi_ngc_sgc_nz.py

# specify where your nz files are:
python combine_desi_ngc_sgc_nz.py --data-dir /path/to/nzfiles --out-dir nz_combined

# custom pairs:
python combine_desi_ngc_sgc_nz.py --pair "BGS=BGS_ANY_NGC_nz.txt,BGS_ANY_SGC_nz.txt" \
                                  --pair "LRG=LRG_NGC_nz.txt,LRG_SGC_nz.txt"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_PAIRS = {
    "BGS": ("BGS_ANY_NGC_nz.txt", "BGS_ANY_SGC_nz.txt"),
    "LRG": ("LRG_NGC_nz.txt", "LRG_SGC_nz.txt"),
    "ELG": ("ELG_LOPnotqso_NGC_nz.txt", "ELG_LOPnotqso_SGC_nz.txt"),
    "QSO": ("QSO_NGC_nz.txt", "QSO_SGC_nz.txt"),
}


def is_html_error(path: Path) -> bool:
    """Catch accidental downloads of HTML error pages (e.g. 504 gateway time-out)."""
    try:
        start = path.read_text(errors="ignore")[:400].lstrip().lower()
    except Exception:
        return False
    return start.startswith("<html") or "gateway time-out" in start


def parse_effective_area_deg2(path: Path) -> Optional[float]:
    """Parse '#effective area is ... square degrees' from the header (if present)."""
    eff = None
    area = None
    with path.open("r") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            m = re.search(r"#\s*effective area is\s*([0-9.]+)\s*square degrees", line, re.I)
            if m:
                eff = float(m.group(1))
            m2 = re.search(r"#\s*area is\s*([0-9.]+)\s*square degrees", line, re.I)
            if m2:
                area = float(m2.group(1))
    return eff if eff is not None else area


def read_nz_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["zmid", "zlow", "zhigh", "nz", "Nbin", "Vol_bin"],
    )
    # basic sanity check
    if not np.isfinite(df["zmid"]).all():
        raise ValueError(f"{path.name}: could not parse numeric columns (file format mismatch?)")
    return df


def combine_ngc_sgc(ngc_path: Path, sgc_path: Path) -> pd.DataFrame:
    if is_html_error(ngc_path) or is_html_error(sgc_path):
        raise ValueError(
            f"HTML error page detected in {ngc_path.name} or {sgc_path.name} "
            "(common if a download returns '504 Gateway Time-out'). Re-download those files."
        )

    df1 = read_nz_table(ngc_path).copy()
    df2 = read_nz_table(sgc_path).copy()

    key = ["zmid", "zlow", "zhigh"]
    m = df1.merge(df2, on=key, suffixes=("_ngc", "_sgc"), how="inner")

    # fallback if zlow/zhigh differ slightly but zmid matches
    if len(m) == 0:
        m = df1.merge(df2, on=["zmid"], suffixes=("_ngc", "_sgc"), how="inner")
        m["zlow"] = m["zlow_ngc"]
        m["zhigh"] = m["zhigh_ngc"]

    m["Ntot"] = m["Nbin_ngc"] + m["Nbin_sgc"]
    m["Vtot"] = m["Vol_bin_ngc"] + m["Vol_bin_sgc"]
    m["nz"] = m["Ntot"] / m["Vtot"]

    out = (
        m[["zmid", "zlow", "zhigh", "nz", "Ntot", "Vtot"]]
        .sort_values("zmid")
        .reset_index(drop=True)
    )
    return out


def compute_dndz_per_sr(df: pd.DataFrame, eff_area_deg2: float) -> np.ndarray:
    """dN/dz per steradian from counts per bin + survey area."""
    dz = (df["zhigh"] - df["zlow"]).to_numpy()
    # convert deg^2 -> sr
    area_sr = eff_area_deg2 * (np.pi / 180.0) ** 2
    dndz_sr = (df["Ntot"].to_numpy() / dz) / area_sr
    return dndz_sr


def parse_pairs(pairs_list) -> Dict[str, Tuple[str, str]]:
    if not pairs_list:
        return dict(DEFAULT_PAIRS)

    out: Dict[str, Tuple[str, str]] = {}
    for s in pairs_list:
        # NAME=file_ngc,file_sgc
        if "=" not in s or "," not in s:
            raise ValueError(f"Bad --pair '{s}'. Expected NAME=NGC_FILE,SGC_FILE")
        name, rest = s.split("=", 1)
        a, b = rest.split(",", 1)
        out[name.strip()] = (a.strip(), b.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Directory containing *_NGC_nz.txt and *_SGC_nz.txt files.")
    ap.add_argument("--out-dir", default="nz_combined", help="Where to write combined tables + plots.")
    ap.add_argument("--pair", action="append", default=None, help="Custom tracer pair: NAME=NGC_FILE,SGC_FILE. Can repeat.")
    ap.add_argument("--plot", choices=["nz", "dndz", "pz"], default="nz",
                    help="Plot comoving n(z) (nz), dN/dz per sr (dndz), or normalized p(z) (pz).")
    ap.add_argument("--yscale", choices=["linear", "log"], default="log",
                    help="Y-axis scaling for the plot.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(args.pair)

    combined = {}
    meta = {}

    for name, (ngc_file, sgc_file) in pairs.items():
        ngc_path = data_dir / ngc_file
        sgc_path = data_dir / sgc_file
        try:
            df = combine_ngc_sgc(ngc_path, sgc_path)
            combined[name] = df

            eff_ngc = parse_effective_area_deg2(ngc_path)
            eff_sgc = parse_effective_area_deg2(sgc_path)
            eff_total = (eff_ngc or 0.0) + (eff_sgc or 0.0)

            meta[name] = {"eff_deg2_ngc": eff_ngc, "eff_deg2_sgc": eff_sgc, "eff_deg2_total": eff_total}

            # write combined table
            out_path = out_dir / f"{name}_NGCplusSGC_nz.txt"
            with out_path.open("w") as f:
                if eff_total > 0:
                    f.write(f"# effective area (deg^2): {eff_total:.6f}\n")
                f.write("# zmid zlow zhigh nz Ntot Vtot\n")
                df.to_csv(f, sep=" ", index=False, header=False, float_format="%.8g")

            print(f"[ok] {name}: wrote {out_path}")

        except Exception as e:
            meta[name] = {"error": str(e)}
            print(f"[skip] {name}: {e}")

    if not combined:
        raise SystemExit("No tracers combined successfully. Check filenames / formats.")

    # make plot
    fig = plt.figure()
    ax = plt.gca()

    for name, df in combined.items():
        if args.plot == "nz":
            y = df["nz"].to_numpy()
            ax.plot(df["zmid"], y, label=name)
            ax.set_ylabel(r"n(z) = N/V  [$h^3\,\mathrm{Mpc}^{-3}$]")
        else:
            eff = meta[name].get("eff_deg2_total", 0.0) or 0.0
            if eff <= 0:
                print(f"[warn] {name}: no effective area found; cannot make dN/dz or p(z).")
                continue
            dndz = compute_dndz_per_sr(df, eff)
            if args.plot == "dndz":
                ax.plot(df["zmid"], dndz, label=name)
                ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}z$  [sr$^{-1}$]")
            else:  # pz
                # normalize to integrate to 1
                z = df["zmid"].to_numpy()
                norm = np.trapz(dndz, z)
                pz = dndz / norm if norm > 0 else dndz * np.nan
                ax.plot(z, pz, label=name)
                ax.set_ylabel(r"$p(z)$ (normalized)")

    ax.set_xlabel("z")
    ax.set_yscale(args.yscale)
    ax.legend()
    ax.set_title("DESI DR1 NGC+SGC combined")
    fig.tight_layout()

    plot_path = out_dir / f"desi_dr1_{args.plot}_combined.png"
    fig.savefig(plot_path, dpi=200)
    print(f"[ok] wrote {plot_path}")


if __name__ == "__main__":
    main()
