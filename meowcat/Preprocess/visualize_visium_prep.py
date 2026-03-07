#!/usr/bin/env python3
"""
visualize_visium_prep.py

QC visualization for Visium prepared inputs.  Overlays spots from
locs.tsv on the processed H&E image (he.jpg / he.tiff) to verify that
the pixel-size scaling is correct.

Outputs per sample (written to <out_root>/<sample>/visium_viz/):
  celltype_<ct>.png    spots colored by cell-type proportion (Reds) over H&E
  argmax_spots.png     spots colored by dominant cell type (tab20) over H&E

Usage
-----
    python visualize_visium_prep.py \\
        --data_root /path/to/data \\
        --out_root  /path/to/outputs \\
        [--samples  SAMPLE1,SAMPLE2] \\
        [--sample_pattern GBM*]     \\
        [--alpha 0.7] [--dpi 150] [--max_side 4096]
"""

import argparse
import glob
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

def _find_he_image(sample_dir: str) -> str:
    for name in ("he.jpg", "he.jpeg", "he.tiff", "he.tif"):
        p = os.path.join(sample_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No processed H&E image (he.jpg / he.tiff) found in {sample_dir}"
    )


def _load_sample(sample_dir: str):
    """
    Load all Visium prep files for one sample.

    Returns
    -------
    cell_types : list[str]
    df         : DataFrame  [N_spots × K] — cell-type proportions, index=spot
    locs       : DataFrame  [N_spots × {x, y}] — processed-image pixels, index=spot
    radius     : int  — spot radius in processed-image pixels
    he_image   : np.ndarray  [H, W, 3]
    """
    for fn in ("anno-names.txt", "anno_matrix.tsv", "locs.tsv", "radius.txt"):
        if not os.path.exists(os.path.join(sample_dir, fn)):
            raise FileNotFoundError(f"Missing {fn} in {sample_dir}")

    with open(os.path.join(sample_dir, "anno-names.txt")) as f:
        cell_types = [ln.strip() for ln in f if ln.strip()]

    df = pd.read_csv(
        os.path.join(sample_dir, "anno_matrix.tsv"), sep="\t", index_col="spot"
    )
    # keep only columns listed in anno-names.txt, in that order
    df = df[[c for c in cell_types if c in df.columns]]

    locs = pd.read_csv(
        os.path.join(sample_dir, "locs.tsv"), sep="\t", index_col="spot"
    )

    with open(os.path.join(sample_dir, "radius.txt")) as f:
        radius = int(f.read().strip())

    # align on shared spots
    common = locs.index.intersection(df.index)
    locs = locs.loc[common]
    df   = df.loc[common]

    he_path  = _find_he_image(sample_dir)
    he_image = np.array(Image.open(he_path).convert("RGB"))

    return cell_types, df, locs, radius, he_image


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _maybe_downsample(he_image, locs, radius, max_side):
    """Downsample image and scale coordinates / radius if image is too large."""
    H, W  = he_image.shape[:2]
    scale = min(1.0, max_side / max(H, W))
    if scale >= 1.0:
        return he_image, locs, radius, 1.0

    new_W, new_H = int(W * scale), int(H * scale)
    he_small = np.array(
        Image.fromarray(he_image).resize((new_W, new_H), Image.LANCZOS)
    )
    locs_small = locs.copy()
    locs_small["x"] = (locs["x"] * scale).round().astype(int)
    locs_small["y"] = (locs["y"] * scale).round().astype(int)
    radius_small = max(1, int(radius * scale))
    return he_small, locs_small, radius_small, scale


def _make_ax(he_image, dpi):
    H, W = he_image.shape[:2]
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(he_image, interpolation="nearest")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _circles(locs, radius):
    return [Circle((row["x"], row["y"]), radius=radius) for _, row in locs.iterrows()]


def _draw_scalar(ax, locs, values, radius, cmap, norm, alpha):
    """Draw spot circles colored by a scalar proportion value."""
    patches = _circles(locs, radius)
    fc = [cmap(norm(float(values.get(spot, 0.0))))[:3] for spot in locs.index]
    pc = PatchCollection(patches, facecolors=fc, alpha=alpha, edgecolors="none")
    ax.add_collection(pc)
    return pc


def _draw_argmax(ax, locs, argmax_series, radius, palette, alpha):
    """Draw spot circles colored by dominant cell type."""
    patches = _circles(locs, radius)
    fc = [palette[argmax_series[spot]][:3] for spot in locs.index]
    pc = PatchCollection(patches, facecolors=fc, alpha=alpha, edgecolors="none")
    ax.add_collection(pc)


# ---------------------------------------------------------------------------
# Per-sample rendering
# ---------------------------------------------------------------------------

def visualize_sample(sample, sample_dir, out_dir, alpha, dpi, max_side):
    print(f"\n[visualize_visium_prep] {sample}")

    cell_types, df, locs, radius, he_image = _load_sample(sample_dir)
    os.makedirs(out_dir, exist_ok=True)

    he_viz, locs_viz, radius_viz, scale = _maybe_downsample(
        he_image, locs, radius, max_side
    )
    H_raw, W_raw = he_image.shape[:2]
    H_viz, W_viz = he_viz.shape[:2]
    print(
        f"  image {W_raw}×{H_raw} → display {W_viz}×{H_viz}  "
        f"scale={scale:.3f}  radius {radius}→{radius_viz}  spots={len(locs)}"
    )

    palette   = {ct: plt.cm.get_cmap("tab20", len(cell_types))(i)
                 for i, ct in enumerate(cell_types)}
    cmap_prob = plt.cm.Reds
    norm_prob = Normalize(vmin=0, vmax=1)

    # ── 1. Per-cell-type probability maps ────────────────────────────────
    for ct in cell_types:
        if ct not in df.columns:
            continue

        fig, ax = _make_ax(he_viz, dpi)
        _draw_scalar(ax, locs_viz, df[ct], radius_viz, cmap_prob, norm_prob, alpha)

        sm = plt.cm.ScalarMappable(cmap=cmap_prob, norm=norm_prob)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label(ct, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        safe = ct.replace("/", "_").replace(" ", "_").replace(":", "_")
        fig.savefig(
            os.path.join(out_dir, f"celltype_{safe}.png"),
            dpi=dpi, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  celltype_{safe}.png")

    # ── 2. Argmax spot overlay ────────────────────────────────────────────
    argmax_idx    = df.values.argmax(axis=1)
    argmax_series = pd.Series(
        [cell_types[i] for i in argmax_idx], index=df.index
    )

    fig, ax = _make_ax(he_viz, dpi)
    _draw_argmax(ax, locs_viz, argmax_series, radius_viz, palette, alpha)

    handles = [
        mpatches.Patch(facecolor=palette[ct][:3], label=ct)
        for ct in cell_types
    ]
    ax.legend(
        handles=handles, fontsize=5, loc="upper right",
        framealpha=0.7, ncol=max(1, len(cell_types) // 10),
        handlelength=1.0, handleheight=0.8,
    )

    fig.savefig(
        os.path.join(out_dir, "argmax_spots.png"),
        dpi=dpi, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  argmax_spots.png")
    print(f"  → {out_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_root", required=True,
        help="Root directory with per-sample subdirectories",
    )
    p.add_argument(
        "--out_root", required=True,
        help="Output root; figures go to <out_root>/<sample>/visium_viz/",
    )
    p.add_argument(
        "--samples", default=None,
        help="Comma-separated sample names (default: all matching --sample_pattern)",
    )
    p.add_argument(
        "--sample_pattern", default="*",
        help="Glob pattern relative to data_root to find sample dirs (default: '*')",
    )
    p.add_argument(
        "--alpha", type=float, default=0.7,
        help="Spot circle opacity 0–1 (default: 0.7)",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="Output figure DPI (default: 150)",
    )
    p.add_argument(
        "--max_side", type=int, default=4096,
        help="Downsample display image so the longer side ≤ this many pixels "
             "(default: 4096)",
    )
    args = p.parse_args()

    # resolve sample list
    if args.samples:
        samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    else:
        pattern = os.path.join(args.data_root, args.sample_pattern)
        samples = sorted(
            os.path.basename(m)
            for m in glob.glob(pattern)
            if os.path.isdir(m)
        )

    if not samples:
        print(
            f"No samples found under {args.data_root} "
            f"with pattern '{args.sample_pattern}'"
        )
        sys.exit(1)

    print(f"[visualize_visium_prep] {len(samples)} sample(s): {samples}")

    errors = []
    for sample in samples:
        sample_dir = os.path.join(args.data_root, sample)
        out_dir    = os.path.join(args.out_root, sample, "visium_viz")
        try:
            visualize_sample(
                sample, sample_dir, out_dir, args.alpha, args.dpi, args.max_side
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((sample, str(e)))

    if errors:
        print("\n[visualize_visium_prep] Errors:")
        for s, e in errors:
            print(f"  {s}: {e}")
        sys.exit(1)

    print("\n[visualize_visium_prep] All done.")


if __name__ == "__main__":
    main()
