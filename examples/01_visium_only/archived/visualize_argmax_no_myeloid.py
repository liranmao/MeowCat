#!/usr/bin/env python3
"""
Visualize predicted cell-type argmax map excluding Myeloid from argmax computation.

Reads the prediction pickle and anno-names.txt, zeroes out the Myeloid column
before computing argmax, and saves the result.

Output: <out_root>/<sample>/<sample>_predicted_celltype_map_no_Myeloid.png
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib import colormaps as mcm
from PIL import Image

# ── Paths (edit if needed) ──────────────────────────────────────────────
DATA_ROOT = "/project/KidneyHE/01_meowcat_test/01_visium_only/input"
OUT_ROOT  = "/project/KidneyHE/01_meowcat_test/01_visium_only/output"
SAMPLE    = "VIS_P11_LUAD"
PKL_NAME  = "pred_fullgrid_outputs.pkl"
EXCLUDE   = "Myeloid"

sample_dir = os.path.join(DATA_ROOT, SAMPLE)
out_dir    = os.path.join(OUT_ROOT, SAMPLE)
pred_file  = os.path.join(DATA_ROOT,SAMPLE, PKL_NAME)

# ── Load prediction ─────────────────────────────────────────────────────
print(f"Loading {pred_file}")
with open(pred_file, "rb") as f:
    pred = pickle.load(f)

p_map = pred["p_map"].astype(np.float32)  # (H, W, K)
H, W, K = p_map.shape
print(f"p_map shape: {p_map.shape}")

# Cell type names
ctypes = pred.get("ctypes", None)
if ctypes is None:
    anno_path = os.path.join(sample_dir, "anno-names.txt")
    if os.path.exists(anno_path):
        with open(anno_path) as f:
            ctypes = [ln.strip() for ln in f if ln.strip()]
    else:
        ctypes = [f"celltype_{i}" for i in range(K)]
print(f"Cell types ({len(ctypes)}): {ctypes}")

# ── Mask ────────────────────────────────────────────────────────────────
mask_path = os.path.join(sample_dir, "mask", "mask-small.png")
if not os.path.exists(mask_path):
    # Try under data_root directly
    mask_path = os.path.join(DATA_ROOT, SAMPLE, "mask", "mask-small.png")
m = np.array(Image.open(mask_path))
if m.ndim == 3:
    m = m[..., 0]
mask = m > 0
print(f"Mask shape: {mask.shape}, tissue pixels: {mask.sum()}")

# ── Exclude Myeloid and compute argmax ──────────────────────────────────
if EXCLUDE in ctypes:
    exc_idx = ctypes.index(EXCLUDE)
    keep_indices = [i for i in range(K) if i != exc_idx]
    keep_names = [ctypes[i] for i in keep_indices]
    p_map_filtered = p_map[:, :, keep_indices]
    print(f"Excluded '{EXCLUDE}' (index {exc_idx}), remaining: {keep_names}")
else:
    keep_names = ctypes
    p_map_filtered = p_map
    print(f"WARNING: '{EXCLUDE}' not found in cell types, using all")

# Argmax
masked = np.where(mask[..., None], p_map_filtered, np.nan)
filled = np.where(np.isnan(masked), -np.inf, masked)
max_indices = np.argmax(filled, axis=-1)
max_indices[np.all(np.isnan(masked), axis=-1)] = -1

# ── Plot ────────────────────────────────────────────────────────────────
n = len(keep_names)
cmap = mcm.get("tab20")
base = list(getattr(cmap, "colors", []))
if not base:
    base = [cmap(i / max(1, cmap.N - 1)) for i in range(cmap.N)]
colors = [to_rgba(base[i % len(base)]) for i in range(n)]

rgba = np.full((H, W, 4), 1.0, dtype=np.float32)  # white background
for i, col in enumerate(colors):
    rgba[max_indices == i] = col

legend_elems = [Patch(facecolor=col, edgecolor="black", label=ct)
                for ct, col in zip(keep_names, colors)]

plt.figure(figsize=(12, 12))
plt.imshow(rgba, interpolation="nearest")
plt.title(f"Predicted Cell Type Map (excl. {EXCLUDE})", fontsize=16)
plt.axis("off")
plt.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1.0, 0.5),
           title="Cell Type", fontsize=10, title_fontsize=12, frameon=False)
plt.tight_layout()

out_png = os.path.join(out_dir, f"{SAMPLE}_predicted_celltype_map_no_{EXCLUDE}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")
