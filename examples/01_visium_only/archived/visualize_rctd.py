#!/usr/bin/env python3
"""
Visualize RCTD deconvolution results overlaid on H&E images.

Generates THREE sets of visualizations to diagnose coordinate alignment:
  A) "original"  — x=pxl_col, y=pxl_row on processed he.jpg (standard convention)
  B) "swapped"   — x=pxl_row, y=pxl_col on processed he.jpg (swapped convention)
  C) "raw"       — x=pxl_col, y=pxl_row on he_raw.tif using raw coords (ground truth)

Outputs (written to <out_root>/<sample>/rctd_viz/):
  <prefix>_argmax.png          — argmax cell type over H&E
  <prefix>_argmax_no_he.png    — argmax cell type on white background
  <prefix>_summary_panel.png   — all cell types in one figure over H&E

Usage:
    python visualize_rctd.py                       # uses ./config.yaml
    python visualize_rctd.py --config path/to.yaml
    python visualize_rctd.py --samples VIS_P11_LUAD
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_config_paths(config_path):
    """Extract data_root, out_root, sample_pattern, and preprocess settings."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project = cfg.get("project", {})
    preprocess = cfg.get("preprocess", {})
    return {
        "data_root": project.get("data_root", ""),
        "out_root": project.get("out_root", ""),
        "sample_pattern": project.get("sample_pattern", "*"),
        "pixel_size_raw": preprocess.get("pixel_size_raw", None),
        "target_mpp": preprocess.get("target_mpp", 0.5),
    }


def find_he_image(sample_dir):
    for name in ("he.jpg", "he.jpeg", "he.tiff", "he.tif"):
        p = os.path.join(sample_dir, name)
        if os.path.exists(p):
            return p
    return None


def find_raw_image(sample_dir, raw_flag="he_raw"):
    """Find raw H&E image by flag substring."""
    for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        p = os.path.join(sample_dir, raw_flag + ext)
        if os.path.exists(p):
            return p
    return None


def read_tissue_positions(spatial_dir):
    """Read spot positions from Space Ranger spatial/ folder."""
    for name in ("tissue_positions_list.csv", "tissue_positions.csv"):
        path = os.path.join(spatial_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "barcode" in df.columns:
            return df.rename(columns={
                "pxl_row_in_fullres": "pxl_row",
                "pxl_col_in_fullres": "pxl_col",
            })
        else:
            return pd.read_csv(
                path, header=None,
                names=["barcode", "in_tissue", "array_row", "array_col",
                       "pxl_row", "pxl_col"],
            )
    raise FileNotFoundError(f"No tissue positions file in {spatial_dir}")


def load_rctd_and_spatial(sample_dir, pixel_size_raw_cfg, target_mpp):
    """
    Load RCTD output and spatial metadata.

    Returns: cell_types, df_rctd, df_tissue (with pxl_row/pxl_col),
             scale, radius_raw, pixel_size_raw
    """
    rctd_csv = os.path.join(sample_dir, "deconvolution_rctd", "major_prop.csv")
    spatial_dir = os.path.join(sample_dir, "spatial")

    if not os.path.exists(rctd_csv):
        raise FileNotFoundError(f"No RCTD output at {rctd_csv}")
    if not os.path.isdir(spatial_dir):
        raise FileNotFoundError(f"No spatial/ directory in {sample_dir}")

    # RCTD proportions
    df_rctd = pd.read_csv(rctd_csv, index_col=0)
    cell_types = list(df_rctd.columns)
    df_rctd.index.name = "spot"

    # Tissue positions (raw fullres pixels)
    df_pos = read_tissue_positions(spatial_dir)
    df_tissue = df_pos[df_pos["in_tissue"] == 1].copy()
    df_tissue = df_tissue.set_index("barcode")

    # Align barcodes
    common = df_rctd.index.intersection(df_tissue.index)
    print(f"  RCTD spots: {len(df_rctd)}, tissue spots: {len(df_tissue)}, "
          f"common: {len(common)}")
    df_rctd = df_rctd.loc[common]
    df_tissue = df_tissue.loc[common]

    # Scale factor: raw -> processed image
    psr_path = os.path.join(sample_dir, "pixel-size-raw.txt")
    if os.path.exists(psr_path):
        with open(psr_path) as f:
            pixel_size_raw = float(f.read().strip())
    elif pixel_size_raw_cfg is not None:
        pixel_size_raw = float(pixel_size_raw_cfg)
    else:
        raise FileNotFoundError(
            f"No pixel-size-raw.txt in {sample_dir} and no pixel_size_raw in config"
        )

    scale = pixel_size_raw / target_mpp

    # Spot radius (raw pixels)
    sf_path = os.path.join(spatial_dir, "scalefactors_json.json")
    with open(sf_path) as f:
        sf = json.load(f)
    radius_raw = sf["spot_diameter_fullres"] / 2.0

    print(f"  pixel_size_raw={pixel_size_raw}, target_mpp={target_mpp}, "
          f"scale={scale:.4f}, radius_raw={radius_raw:.1f}")

    return cell_types, df_rctd, df_tissue, scale, radius_raw, pixel_size_raw


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def downsample_for_display(he_image, locs_df, radius, max_side=4096):
    H, W = he_image.shape[:2]
    s = min(1.0, max_side / max(H, W))
    if s >= 1.0:
        return he_image, locs_df, radius, 1.0
    new_W, new_H = int(W * s), int(H * s)
    he_small = np.array(
        Image.fromarray(he_image).resize((new_W, new_H), Image.LANCZOS)
    )
    locs_small = locs_df.copy()
    locs_small["x"] = (locs_df["x"] * s).round().astype(int)
    locs_small["y"] = (locs_df["y"] * s).round().astype(int)
    return he_small, locs_small, max(1, int(radius * s)), s


def make_ax(he_image, dpi=150):
    H, W = he_image.shape[:2]
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(he_image, interpolation="nearest")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def render_argmax_on_he(he_viz, locs_viz, argmax_names, cell_types, palette,
                        r_viz, dpi, alpha, title, out_path):
    fig, ax = make_ax(he_viz, dpi)
    patches = [Circle((row["x"], row["y"]), radius=r_viz)
               for _, row in locs_viz.iterrows()]
    fc = [palette[argmax_names[i]][:3] for i in range(len(locs_viz))]
    pc = PatchCollection(patches, facecolors=fc, alpha=alpha, edgecolors="none")
    ax.add_collection(pc)
    handles = [mpatches.Patch(facecolor=palette[ct][:3], label=ct)
               for ct in cell_types]
    ax.legend(handles=handles, fontsize=5, loc="upper right",
              framealpha=0.7, handlelength=1.0, handleheight=0.8)
    ax.set_title(title, fontsize=9, pad=4)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_argmax_no_he(he_shape, locs_viz, argmax_names, cell_types, palette,
                        r_viz, dpi, title, out_path):
    H, W = he_shape[:2]
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_facecolor("white")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    patches = [Circle((row["x"], row["y"]), radius=r_viz)
               for _, row in locs_viz.iterrows()]
    fc = [palette[argmax_names[i]][:3] for i in range(len(locs_viz))]
    pc = PatchCollection(patches, facecolors=fc, alpha=1.0, edgecolors="none")
    ax.add_collection(pc)
    handles = [mpatches.Patch(facecolor=palette[ct][:3], label=ct)
               for ct in cell_types]
    ax.legend(handles=handles, fontsize=5, loc="upper right",
              framealpha=0.7, handlelength=1.0, handleheight=0.8)
    ax.set_title(title, fontsize=9, pad=4)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_summary_panel(he_viz, locs_viz, df_rctd, cell_types, r_viz, alpha,
                         title, out_path):
    cmap = plt.cm.Reds
    norm = Normalize(vmin=0, vmax=1)
    n_ct = len(cell_types)
    ncols = min(4, n_ct)
    nrows = (n_ct + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=100)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ct in enumerate(cell_types):
        ax = axes[i]
        ax.imshow(he_viz, interpolation="nearest")
        ax.set_xlim(0, he_viz.shape[1])
        ax.set_ylim(he_viz.shape[0], 0)
        ax.axis("off")
        vals = df_rctd[ct].loc[locs_viz.index].values
        patches_i = [Circle((row["x"], row["y"]), radius=r_viz)
                     for _, row in locs_viz.iterrows()]
        fc_i = [cmap(norm(v))[:3] for v in vals]
        pc_i = PatchCollection(patches_i, facecolors=fc_i, alpha=alpha,
                               edgecolors="none")
        ax.add_collection(pc_i)
        ax.set_title(ct, fontsize=10)

    for j in range(n_ct, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-variant rendering
# ---------------------------------------------------------------------------

def render_variant(prefix, he_image, locs_df, radius, df_rctd, cell_types,
                   out_dir, alpha, dpi, max_side):
    """Render argmax (with & without H&E) and summary panel for one coordinate variant."""
    he_viz, locs_viz, r_viz, dscale = downsample_for_display(
        he_image, locs_df, radius, max_side
    )
    print(f"  [{prefix}] display: {he_viz.shape[1]}x{he_viz.shape[0]}, "
          f"radius={r_viz}, spots={len(locs_viz)}")

    palette = {ct: plt.cm.tab20(i / max(1, len(cell_types) - 1))
               for i, ct in enumerate(cell_types)}

    argmax_idx = df_rctd.values.argmax(axis=1)
    argmax_names = [cell_types[i] for i in argmax_idx]

    # Argmax on H&E
    render_argmax_on_he(
        he_viz, locs_viz, argmax_names, cell_types, palette, r_viz, dpi, alpha,
        f"RCTD argmax — {prefix}",
        os.path.join(out_dir, f"{prefix}_argmax.png"),
    )
    print(f"  saved: {prefix}_argmax.png")

    # Argmax on white
    render_argmax_no_he(
        he_viz.shape, locs_viz, argmax_names, cell_types, palette, r_viz, dpi,
        f"RCTD argmax (no H&E) — {prefix}",
        os.path.join(out_dir, f"{prefix}_argmax_no_he.png"),
    )
    print(f"  saved: {prefix}_argmax_no_he.png")

    # Argmax excluding Myeloid (on H&E and no H&E)
    exclude_ct = "Myeloid"
    keep_cols = [ct for ct in cell_types if ct != exclude_ct]
    if keep_cols and exclude_ct in cell_types:
        df_no_mye = df_rctd[keep_cols]
        argmax_idx_no_mye = df_no_mye.values.argmax(axis=1)
        argmax_names_no_mye = [keep_cols[i] for i in argmax_idx_no_mye]

        render_argmax_on_he(
            he_viz, locs_viz, argmax_names_no_mye, keep_cols, palette, r_viz, dpi, alpha,
            f"RCTD argmax excl. {exclude_ct} — {prefix}",
            os.path.join(out_dir, f"{prefix}_argmax_no_{exclude_ct}.png"),
        )
        print(f"  saved: {prefix}_argmax_no_{exclude_ct}.png")

        render_argmax_no_he(
            he_viz.shape, locs_viz, argmax_names_no_mye, keep_cols, palette, r_viz, dpi,
            f"RCTD argmax excl. {exclude_ct} (no H&E) — {prefix}",
            os.path.join(out_dir, f"{prefix}_argmax_no_{exclude_ct}_no_he.png"),
        )
        print(f"  saved: {prefix}_argmax_no_{exclude_ct}_no_he.png")

    # Summary panel
    render_summary_panel(
        he_viz, locs_viz, df_rctd, cell_types, r_viz, alpha,
        f"RCTD — {prefix}",
        os.path.join(out_dir, f"{prefix}_summary_panel.png"),
    )
    print(f"  saved: {prefix}_summary_panel.png")


# ---------------------------------------------------------------------------
# Per-sample orchestration
# ---------------------------------------------------------------------------

def visualize_one_sample(sample, sample_dir, out_dir,
                         pixel_size_raw_cfg, target_mpp,
                         alpha=0.7, dpi=150, max_side=4096):
    print(f"\n{'='*60}")
    print(f"[visualize_rctd] Sample: {sample}")
    print(f"{'='*60}")

    cell_types, df_rctd, df_tissue, scale, radius_raw, pixel_size_raw = \
        load_rctd_and_spatial(sample_dir, pixel_size_raw_cfg, target_mpp)
    os.makedirs(out_dir, exist_ok=True)

    # ── A) Original convention: x=pxl_col, y=pxl_row on processed he.jpg ──
    he_path = find_he_image(sample_dir)
    if he_path:
        he_proc = np.array(Image.open(he_path).convert("RGB"))
        print(f"  processed H&E: {he_path}  shape={he_proc.shape}")

        locs_orig = pd.DataFrame({
            "x": (df_tissue["pxl_col"].values * scale).round().astype(int),
            "y": (df_tissue["pxl_row"].values * scale).round().astype(int),
        }, index=df_rctd.index)
        radius_proc = int(radius_raw * scale)

        render_variant("original", he_proc, locs_orig, radius_proc,
                       df_rctd, cell_types, out_dir, alpha, dpi, max_side)
    else:
        print("  WARNING: no processed H&E found, skipping original variant")

    # ── B) Swapped convention: x=pxl_row, y=pxl_col on processed he.jpg ──
    if he_path:
        locs_swap = pd.DataFrame({
            "x": (df_tissue["pxl_row"].values * scale).round().astype(int),
            "y": (df_tissue["pxl_col"].values * scale).round().astype(int),
        }, index=df_rctd.index)

        render_variant("swapped", he_proc, locs_swap, radius_proc,
                       df_rctd, cell_types, out_dir, alpha, dpi, max_side)

    # ── C) Ground truth: x=pxl_col, y=pxl_row on raw he_raw image ─────────
    raw_path = find_raw_image(sample_dir)
    if raw_path:
        print(f"  raw H&E: {raw_path}")
        he_raw = np.array(Image.open(raw_path).convert("RGB"))
        print(f"  raw H&E shape={he_raw.shape}")

        locs_raw = pd.DataFrame({
            "x": df_tissue["pxl_col"].values.astype(int),
            "y": df_tissue["pxl_row"].values.astype(int),
        }, index=df_rctd.index)
        radius_raw_px = int(radius_raw)

        render_variant("raw", he_raw, locs_raw, radius_raw_px,
                       df_rctd, cell_types, out_dir, alpha, dpi, max_side)
    else:
        print("  WARNING: no raw H&E found, skipping raw variant")

    # ── Print proportion statistics ────────────────────────────────────
    print(f"\n  Cell-type proportion statistics (mean +/- std):")
    for ct in cell_types:
        vals = df_rctd[ct].values
        print(f"    {ct:20s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"max={vals.max():.4f}  >0.5: {(vals > 0.5).sum():5d} spots")

    argmax_idx = df_rctd.values.argmax(axis=1)
    argmax_names = [cell_types[i] for i in argmax_idx]
    argmax_counts = pd.Series(argmax_names).value_counts()
    print(f"\n  Argmax cell-type distribution:")
    for ct in cell_types:
        count = argmax_counts.get(ct, 0)
        print(f"    {ct:20s}: {count:6d} spots ({100*count/len(argmax_names):.1f}%)")

    print(f"\n  All outputs -> {out_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                   help="Path to MeowCat config YAML (default: ./config.yaml)")
    p.add_argument("--samples", default=None,
                   help="Comma-separated sample names (default: auto-discover)")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max_side", type=int, default=4096)
    args = p.parse_args()

    cfg = load_config_paths(args.config)
    data_root = cfg["data_root"]
    out_root = cfg["out_root"]

    # Discover samples
    if args.samples:
        samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    else:
        pattern = os.path.join(data_root, cfg["sample_pattern"])
        samples = sorted(
            os.path.basename(m) for m in glob.glob(pattern) if os.path.isdir(m)
        )

    if not samples:
        print(f"No samples found under {data_root} with pattern '{cfg['sample_pattern']}'")
        sys.exit(1)

    print(f"[visualize_rctd] Samples: {samples}")
    print(f"[visualize_rctd] data_root: {data_root}")
    print(f"[visualize_rctd] out_root:  {out_root}")

    for sample in samples:
        sample_dir = os.path.join(data_root, sample)
        out_dir = os.path.join(out_root, sample, "rctd_viz")
        try:
            visualize_one_sample(
                sample, sample_dir, out_dir,
                cfg["pixel_size_raw"], cfg["target_mpp"],
                args.alpha, args.dpi, args.max_side,
            )
        except Exception as e:
            print(f"  ERROR for {sample}: {e}")
            import traceback
            traceback.print_exc()

    print("\n[visualize_rctd] Done.")


if __name__ == "__main__":
    main()
