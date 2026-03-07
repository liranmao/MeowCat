#!/usr/bin/env python3
import glob
from pathlib import Path
import pandas as pd
import argparse


def get_target_order(base_dir: Path) -> list:
    """
    Find one major_prop.csv under base_dir/P*/deconvolution_rctd/ and use its colnames as the target order.
    """
    candidates = sorted(base_dir.glob("P*/deconvolution_rctd/major_prop.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No major_prop.csv found under {base_dir}/P*/deconvolution_rctd/"
        )

    example_file = candidates[0]
    print(f"[INFO] Using column order from: {example_file}", flush=True)

    # Read only once; we just need the column names
    df_example = pd.read_csv(example_file, index_col=0)
    target_order = list(df_example.columns)

    return target_order



# def csv_to_tsv(df: pd.DataFrame, target_order: list) -> pd.DataFrame:
#     """
#     Convert the CSV-format dataframe into TSV format:
#       - add 'spot' column from index
#       - ensure target_order columns exist; missing columns filled with 0
#       - preserve any extra columns (append them after target_order)
#     """
#     if df.index.name is None or df.index.name == "":
#         df.index.name = "spot"
#     df = df.reset_index()

#     # Add missing cols as 0
#     for col in target_order:
#         if col not in df.columns:
#             df[col] = 0.0

#     # Reorder: target first, then extras
#     target_existing = [c for c in target_order if c in df.columns]
#     extras = [c for c in df.columns if c not in target_existing]
#     ordered = target_existing + extras
#     return df[ordered]

def csv_to_tsv(df: pd.DataFrame, target_order: list) -> pd.DataFrame:
    """
    Convert the CSV-format dataframe into TSV format:
      - add 'spot' column from index
      - ensure target_order columns exist; missing columns filled with 0
      - preserve any extra columns (append them after target_order)
    """
    if df.index.name is None or df.index.name == "":
        df.index.name = "spot"
    df = df.reset_index()

    # Add missing cols as 0
    for col in target_order:
        if col not in df.columns:
            df[col] = 0.0

    # Reorder: spot first (if present), then target_order, then extras
    first_cols = ["spot"] if "spot" in df.columns else []
    target_existing = [c for c in target_order if c in df.columns]
    extras = [c for c in df.columns if c not in first_cols + target_existing]

    ordered = first_cols + target_existing + extras
    return df[ordered]



def process_sample(csv_path: Path, out_base: Path, target_order: list):
    """
    For each major_prop.csv at:
        {base_dir}/P*/deconvolution_rctd/major_prop.csv
      - sample_name = P* directory name (e.g. P8_LUAD)
      - out_dir = {base_dir}/7_new_sc_data/{sample_name}
      - write:
          - anno_matrix.tsv
          - anno-names.txt (one cell type per line, from RCTD columns)
    """
    # major_prop.csv is at: base_dir / P* / deconvolution_rctd / major_prop.csv
    sample_name = csv_path.parent.parent.name  # the P* directory name
    out_dir = out_base / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    anno_path = out_dir / "anno_matrix.tsv"
    anno_names_path = out_dir / "anno-names.txt"

    print(f"[INFO] Processing sample: {sample_name}", flush=True)
    print(f"       Input CSV: {csv_path}", flush=True)
    print(f"       Output dir: {out_dir}", flush=True)

    # Read RCTD major_prop.csv
    df = pd.read_csv(csv_path, index_col=0)

    # ---- NEW: write anno-names.txt from RCTD column names ----
    with open(anno_names_path, "w") as f:
        for ct in df.columns:
            f.write(f"{ct}\n")
    print(f"       Wrote {anno_names_path.name}", flush=True)
    # ---------------------------------------------------------

    # Reformat and write anno_matrix.tsv
    tsv_df = csv_to_tsv(df, target_order)
    tsv_df.to_csv(anno_path, sep="\t", index=False)
    print(f"       Wrote new {anno_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Reformat RCTD deconvolution CSVs into anno_matrix.tsv format"
    )
    parser.add_argument(
        "base_dir",
        help="Base directory (e.g. /project/KidneyHE/data_lung/) where P*/ and P*_LUAD/ live",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_base = base_dir / "7_new_sc_data"

    print(f"[INFO] Base dir: {base_dir}", flush=True)
    print(f"[INFO] Output base dir: {out_base}", flush=True)

    target_order = get_target_order(base_dir)

    # RCTD files are now at: base_dir / P* / deconvolution_rctd / major_prop.csv
    csv_files = list(base_dir.glob("P*/deconvolution_rctd/major_prop.csv"))
    if not csv_files:
        print(f"[WARNING] No major_prop.csv files found under {base_dir}/P*/deconvolution_rctd/", flush=True)
        return

    print(f"[INFO] Found {len(csv_files)} CSV files to process", flush=True)
    for csv_path in csv_files:
        process_sample(csv_path, out_base, target_order)

    print("[INFO] Done!", flush=True)


if __name__ == "__main__":
    main()
