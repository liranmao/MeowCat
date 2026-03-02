#!/usr/bin/env python3
"""
prepare_visium_inputs.py

Prepare all Visium-specific input files required by the MeowCat pipeline.

Must be run AFTER:
  - RCTD deconvolution  (meowcat rctd)
  - get_pixel_size.py has written pixel-size-raw.txt
    (first sub-step of meowcat preprocess, OR set manually)

Reads RCTD output and Visium spatial metadata; writes into <sample_dir>/:

  anno-names.txt     cell-type names from RCTD columns, one per line
  anno_matrix.tsv    spot × cell-type proportions  (with leading 'spot' column)
  locs-raw.tsv       spot x/y in raw fullres pixel coordinates
  locs.tsv           spot x/y scaled to processed image (he.jpg) pixels
  radius-raw.txt     spot radius in raw image pixels
                     = spot_diameter_fullres / 2  (from scalefactors_json.json)
  radius.txt         spot radius in processed image pixels
                     = int(radius_raw * pixel_size_raw / target_mpp)
  pixel-size.txt     target MPP  (= target_mpp config value, default 0.5)

Coordinate conventions
----------------------
locs-raw.tsv stores raw fullres pixel coordinates (pxl_col_in_fullres = x,
pxl_row_in_fullres = y).

locs.tsv stores those same coordinates scaled to the processed image
(he.jpg) resolution:  x_scaled = round(x_raw * pixel_size_raw / target_mpp).

radius.txt stores the spot radius in processed he.jpg pixels.  The batch
preparation code further divides by 16 to map to the UNI feature grid.

Non-Visium samples (no RCTD output or no spatial/ folder) are skipped
gracefully so this script can safely run as part of meowcat preprocess
for any sample type.
"""

import argparse
import json
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_tissue_positions(spatial_dir: str) -> pd.DataFrame:
    """
    Read spot positions from a Space Ranger spatial/ folder.

    Handles two formats:
      - tissue_positions_list.csv  (Space Ranger < 2.0, no header)
      - tissue_positions.csv       (Space Ranger >= 2.0, has header)

    Returns a DataFrame with columns:
        barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
    """
    for name in ("tissue_positions_list.csv", "tissue_positions.csv"):
        path = os.path.join(spatial_dir, name)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        if "barcode" in df.columns:
            # New format already has a header row
            return df.rename(columns={
                "pxl_row_in_fullres": "pxl_row",
                "pxl_col_in_fullres": "pxl_col",
            })
        else:
            # Old format: no header
            # columns: barcode, in_tissue, array_row, array_col,
            #          pxl_row_in_fullres, pxl_col_in_fullres
            df = pd.read_csv(
                path, header=None,
                names=["barcode", "in_tissue", "array_row", "array_col",
                       "pxl_row", "pxl_col"],
            )
            return df

    raise FileNotFoundError(
        f"No tissue positions file found in {spatial_dir} "
        f"(tried tissue_positions_list.csv and tissue_positions.csv)"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sample_dir", required=True,
        help="Path to the sample directory (must contain spatial/ and "
             "deconvolution_rctd/major_prop.csv)",
    )
    p.add_argument(
        "--target_mpp", type=float, default=0.5,
        help="Target microns-per-pixel used for image rescaling (default: 0.5)",
    )
    args = p.parse_args()

    sample_dir = args.sample_dir
    target_mpp = args.target_mpp

    # ------------------------------------------------------------------
    # Guard: skip gracefully if this is not a Visium sample
    # ------------------------------------------------------------------
    rctd_csv    = os.path.join(sample_dir, "deconvolution_rctd", "major_prop.csv")
    spatial_dir = os.path.join(sample_dir, "spatial")

    if not os.path.exists(rctd_csv) or not os.path.isdir(spatial_dir):
        print(
            f"[prepare_visium_inputs] Skipping {sample_dir}: "
            f"not a Visium sample "
            f"(missing deconvolution_rctd/major_prop.csv or spatial/ folder)."
        )
        sys.exit(0)

    print(f"[prepare_visium_inputs] Processing: {sample_dir}")

    # ------------------------------------------------------------------
    # 1. Read pixel-size-raw.txt  (written by get_pixel_size.py, or manual)
    # ------------------------------------------------------------------
    pixel_size_raw_path = os.path.join(sample_dir, "pixel-size-raw.txt")
    if not os.path.exists(pixel_size_raw_path):
        print(
            f"ERROR: pixel-size-raw.txt not found in {sample_dir}.\n"
            f"       Run 'meowcat preprocess' (or get_pixel_size.py) first, "
            f"or create the file manually with the raw MPP value."
        )
        sys.exit(1)

    with open(pixel_size_raw_path) as f:
        pixel_size_raw = float(f.read().strip())

    scale = pixel_size_raw / target_mpp
    print(f"  pixel_size_raw = {pixel_size_raw:.6f} mpp")
    print(f"  target_mpp     = {target_mpp}")
    print(f"  scale          = {scale:.6f}  (processed / raw)")

    # ------------------------------------------------------------------
    # 2. RCTD output → anno-names.txt  +  anno_matrix.tsv
    # ------------------------------------------------------------------
    df_rctd    = pd.read_csv(rctd_csv, index_col=0)
    cell_types = list(df_rctd.columns)

    with open(os.path.join(sample_dir, "anno-names.txt"), "w") as f:
        for ct in cell_types:
            f.write(f"{ct}\n")
    print(f"  Wrote anno-names.txt  ({len(cell_types)} cell types: {cell_types})")

    df_rctd.index.name = "spot"
    df_anno = df_rctd.reset_index()
    df_anno.to_csv(
        os.path.join(sample_dir, "anno_matrix.tsv"), sep="\t", index=False
    )
    print(
        f"  Wrote anno_matrix.tsv "
        f"({len(df_anno)} spots × {len(cell_types)} cell types)"
    )

    # ------------------------------------------------------------------
    # 3. Visium spatial positions → locs-raw.tsv  +  locs.tsv
    #
    #    locs-raw.tsv : spot x/y in raw fullres pixel coordinates
    #    locs.tsv     : spot x/y scaled to processed image (he.jpg) pixels
    #                   = locs_raw * (pixel_size_raw / target_mpp)
    # ------------------------------------------------------------------
    df_pos    = _read_tissue_positions(spatial_dir)
    df_tissue = df_pos[df_pos["in_tissue"] == 1].copy()

    df_locs_raw = pd.DataFrame({
        "spot": df_tissue["barcode"].values,
        "x":    df_tissue["pxl_col"].values.astype(int),  # pxl_col_in_fullres
        "y":    df_tissue["pxl_row"].values.astype(int),  # pxl_row_in_fullres
    })
    df_locs_raw.to_csv(
        os.path.join(sample_dir, "locs-raw.tsv"), sep="\t", index=False
    )
    print(
        f"  Wrote locs-raw.tsv    "
        f"({len(df_locs_raw)} in-tissue spots, raw fullres coords)"
    )

    # Scale x, y to processed image resolution; spot column is unchanged
    df_locs = df_locs_raw.copy()
    df_locs["x"] = (df_locs_raw["x"] * scale).round().astype(int)
    df_locs["y"] = (df_locs_raw["y"] * scale).round().astype(int)
    df_locs.to_csv(
        os.path.join(sample_dir, "locs.tsv"), sep="\t", index=False
    )
    print(
        f"  Wrote locs.tsv        "
        f"({len(df_locs)} in-tissue spots, processed-image coords)"
    )

    # ------------------------------------------------------------------
    # 4. Spot radius  →  radius-raw.txt  +  radius.txt
    #
    #    radius-raw.txt : radius in raw image pixels
    #    radius.txt     : radius in processed image (he.jpg) pixels
    #                     batch prep code further divides by 16 to convert
    #                     to the UNI feature-grid pixel space
    # ------------------------------------------------------------------
    scalefactors_path = os.path.join(spatial_dir, "scalefactors_json.json")
    if not os.path.exists(scalefactors_path):
        print(f"ERROR: scalefactors_json.json not found at {scalefactors_path}")
        sys.exit(1)

    with open(scalefactors_path) as f:
        sf = json.load(f)

    radius_raw    = sf["spot_diameter_fullres"] / 2.0
    radius_scaled = int(radius_raw * scale)

    with open(os.path.join(sample_dir, "radius-raw.txt"), "w") as f:
        f.write(f"{radius_raw:.6f}\n")
    print(f"  Wrote radius-raw.txt  ({radius_raw:.4f} raw pixels)")

    with open(os.path.join(sample_dir, "radius.txt"), "w") as f:
        f.write(f"{radius_scaled}\n")
    print(
        f"  Wrote radius.txt      "
        f"({radius_scaled} processed-image pixels, "
        f"÷16 → {radius_scaled // 16} feature-grid pixels)"
    )

    # ------------------------------------------------------------------
    # 5. pixel-size.txt  (target MPP, companion to pixel-size-raw.txt)
    # ------------------------------------------------------------------
    with open(os.path.join(sample_dir, "pixel-size.txt"), "w") as f:
        f.write(f"{target_mpp:.10f}\n")
    print(f"  Wrote pixel-size.txt  ({target_mpp} mpp)")

    print(f"\n[prepare_visium_inputs] Done.")


if __name__ == "__main__":
    main()
