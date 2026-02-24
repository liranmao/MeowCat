#!/usr/bin/env python3
"""
Process a single sample:
- Load prediction pickle (expects keys: 'z_map', 'p_map', 'ctypes' optional)
- Load mask image (SAMPLE/mask/mask-small.png)
- KMeans on z_map within mask, save full cluster map (white background outside mask)
- (Optional) per-cluster highlight images
- Masked per-cell-type intensity maps with percentile scaling (default 5–95)
- Predicted cell-type argmax map + legend

All outputs are saved under: <out_root>/<SAMPLE>/...

Author: you :)
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib import colormaps as mcm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



# ---- Optional: SciPy for nicer outlines (not strictly required) ----
try:
    from scipy.ndimage import binary_dilation, binary_erosion
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---- Robust image loader (no external utils required) ----
def load_image(path: str) -> np.ndarray:
    """
    Load an image as a numpy array. Returns an array shaped (H, W) for single-channel
    or (H, W, C) for multi-channel. Raises if missing.
    """
    from PIL import Image
    img = Image.open(path)
    return np.array(img)

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def safe_name(s: str) -> str:
    return s.replace("/", "-").replace("\\", "-").replace(" ", "_")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ------------------
# Visualization utils
# ------------------
def build_categorical_colors(n: int, base_palette: str = "tab20") -> List[Tuple[float, float, float, float]]:
    cmap = mcm.get(base_palette)
    # use discrete .colors if present, else sample evenly
    base = list(getattr(cmap, "colors", []))
    if not base:
        base = [cmap(i / max(1, cmap.N - 1)) for i in range(cmap.N)]
    base = [to_rgba(c) for c in base]
    return [base[i % len(base)] for i in range(n)]

def save_cluster_map(lab_map: np.ndarray,
                     n_clusters: int,
                     out_png: str,
                     title: str = "",
                     base_palette: str = "tab20") -> None:
    H, W = lab_map.shape
    colors = build_categorical_colors(n_clusters, base_palette=base_palette)

    rgba = np.full((H, W, 4), 1.0, dtype=np.float32)  # white background
    for k in range(n_clusters):
        rgba[lab_map == k] = colors[k]

    legend_elems = [Patch(facecolor=colors[k], edgecolor="black", label=f"Cluster {k}")
                    for k in range(n_clusters)]

    plt.figure(figsize=(12, 12))
    plt.imshow(rgba, interpolation="nearest")
    if title:
        plt.title(title, fontsize=16)
    plt.axis("off")
    plt.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1.0, 0.5),
               title="Clusters", fontsize=10, title_fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_single_cluster_highlight(lab_map: np.ndarray,
                                  cid: int,
                                  out_png: str,
                                  mask: Optional[np.ndarray] = None,
                                  alpha_bg: float = 0.12,
                                  base_palette: str = "tab20",
                                  title_prefix: str = "Cluster") -> None:
    H, W = lab_map.shape
    tissue_mask = (lab_map >= 0) if mask is None else mask.astype(bool)
    sel = (lab_map == cid) & tissue_mask

    # Canvas: white outside tissue
    bg = np.ones((H, W, 3), dtype=np.float32)

    # Overlays
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    other_tissue = tissue_mask & (~sel)
    rgba[other_tissue, :3] = 0.0
    rgba[other_tissue, 3] = alpha_bg

    colors = build_categorical_colors(int(lab_map.max() + 1), base_palette=base_palette)
    col = np.array(colors[cid][:3], dtype=np.float32)
    rgba[sel, :3] = col
    rgba[sel, 3] = 1.0

    plt.figure(figsize=(10, 10))
    plt.imshow(bg, interpolation="nearest")
    plt.imshow(rgba, interpolation="nearest")

    # Optional outline
    if _HAS_SCIPY and np.any(sel):
        dil = binary_dilation(sel)
        ero = binary_erosion(sel)
        edge = (dil ^ ero)
        edge_img = np.zeros((H, W, 4), dtype=np.float32)
        edge_img[edge] = [0, 0, 0, 1.0]
        plt.imshow(edge_img, interpolation="nearest")

    plt.title(f"{title_prefix} {cid}", fontsize=16)
    plt.axis("off")
    legend_elems = [Patch(facecolor=col, edgecolor='black', label=f"{title_prefix} {cid}")]
    plt.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1.02, 0.5),
               title="Selected", fontsize=10, title_fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_masked_intensity(img: np.ndarray,
                          mask: np.ndarray,
                          out_png: str,
                          title: str,
                          p_lo: float = 5.0,
                          p_hi: float = 95.0) -> None:
    masked = np.where(mask, img, np.nan)
    vmin = float(np.nanpercentile(masked, p_lo))
    vmax = float(np.nanpercentile(masked, p_hi))

    plt.figure(figsize=(10, 8))
    plt.imshow(masked, cmap="turbo", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(label="Intensity")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_argmax_celltype_map(probs: np.ndarray,
                             cell_types: List[str],
                             mask: np.ndarray,
                             out_png: str,
                             base_palette: str = "tab20") -> None:
    # probs: (H, W, K)
    H, W, K = probs.shape
    masked = np.where(mask[..., None], probs, np.nan)
    # argmax with NaNs handled: set NaN rows to -inf so they don't win
    filled = np.where(np.isnan(masked), -np.inf, masked)
    max_indices = np.argmax(filled, axis=-1)
    # fully masked -> -1
    max_indices[np.all(np.isnan(masked), axis=-1)] = -1

    colors = build_categorical_colors(len(cell_types), base_palette=base_palette)
    rgba = np.full((H, W, 4), 1.0, dtype=np.float32)
    for i, col in enumerate(colors):
        rgba[max_indices == i] = col

    legend_elems = [Patch(facecolor=col, edgecolor='black', label=ct)
                    for ct, col in zip(cell_types, colors)]

    plt.figure(figsize=(12, 12))
    plt.imshow(rgba, interpolation="nearest")
    plt.title("Predicted Cell Type Map", fontsize=16)
    plt.axis("off")
    plt.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1.0, 0.5),
               title="Cell Type", fontsize=10, title_fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------
# Core processing
# ------------------
def run_one_sample(
    data_root: str,
    sample: str,
    out_root: str,
    pkl_name: Optional[str] = None,
    pkl_path: Optional[str] = None,
    data_root_ori: Optional[str] = None,
    n_clusters: int = 6,
    pca_comp: int = 100,
    random_seed: int = 0,
    save_highlights: bool = False,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
) -> None:
    sample_dir = os.path.join(data_root, sample)
    if data_root_ori is None:
        data_root_ori = data_root
    sample_dir_ori = os.path.join(data_root_ori, sample)

    # Resolve prediction file
    if pkl_path is not None:
        pred_file = pkl_path
    else:
        if not pkl_name:
            raise ValueError("Provide either --pkl-path or --pkl-name.")
        pred_file = os.path.join(sample_dir, pkl_name)

    if not os.path.isfile(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    # Load prediction
    with open(pred_file, "rb") as f:
        pred = pickle.load(f)

    # Required tensors
    z_map = pred["z_map"].astype(np.float32)   # (H, W, D)
    p_map = pred["p_map"].astype(np.float32)   # (H, W, K)
    H, W, D = z_map.shape
    K = p_map.shape[-1]

    # Cell type names
    ctypes = pred.get("ctypes", None)
    if ctypes is None:
        anno_path = os.path.join(sample_dir, "anno-names.txt")
        if os.path.exists(anno_path):
            ctypes = read_lines(anno_path)
        else:
            ctypes = [f"celltype_{i}" for i in range(K)]
    if len(ctypes) != K:
        # be defensive
        ctypes = [f"{ctypes[i] if i < len(ctypes) else f'celltype_{i}'}" for i in range(K)]

    # Mask
    mask_path = os.path.join(sample_dir_ori, "mask", "mask-small.png")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    m = load_image(mask_path)
    if m.ndim == 3:
        # If RGB/RGBA mask, reduce to single channel
        m = m[..., 0]
    mask = m > 0
    if mask.shape != (H, W):
        raise ValueError(f"Mask shape {mask.shape} does not match z_map shape {(H, W)}")

    # Prepare output folders
    out_sample = os.path.join(out_root, sample)
    out_clusters = os.path.join(out_sample, "clusters")
    out_highlights = os.path.join(out_sample, "cluster_highlights")
    out_intensity = os.path.join(out_sample, "celltype_intensity_percentiles")
    ensure_dir(out_sample)
    ensure_dir(out_clusters)
    if save_highlights:
        ensure_dir(out_highlights)
    ensure_dir(out_intensity)

    # -------- Clustering on z_map within mask --------
    Z_full = z_map.reshape(-1, D)
    m_flat = mask.reshape(-1)
    idx = np.where(m_flat)[0]
    Z = Z_full[idx]

    scaler = StandardScaler(with_mean=True, with_std=True)
    Zs = scaler.fit_transform(Z)

    pca = PCA(n_components=min(int(pca_comp), Zs.shape[1]), svd_solver="randomized", random_state=random_seed)
    Zp = pca.fit_transform(Zs)

    try:
        km = KMeans(n_clusters=int(n_clusters), n_init="auto", random_state=random_seed)
    except TypeError:
        # for older scikit-learn versions
        km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=random_seed)

    labels_masked = km.fit_predict(Zp)
    lab_flat = np.full(H * W, fill_value=-1, dtype=np.int32)
    lab_flat[idx] = labels_masked
    lab_map = lab_flat.reshape(H, W)

    # Save overall cluster map
    cluster_map_png = os.path.join(out_clusters, f"{sample}_kmeans_k{n_clusters}.png")
    save_cluster_map(
        lab_map,
        n_clusters=n_clusters,
        out_png=cluster_map_png,
        title=f"{sample} – KMeans clusters on z_map (k={n_clusters})",
        base_palette="tab20",
    )

    # Optional per-cluster highlight images
    if save_highlights:
        for cid in range(n_clusters):
            out_png = os.path.join(out_highlights, f"{sample}_cluster_{cid:02d}.png")
            save_single_cluster_highlight(
                lab_map,
                cid,
                out_png=out_png,
                mask=mask,
                alpha_bg=0.10,
                base_palette="tab20",
                title_prefix="Cluster"
            )

    # -------- Masked intensity maps for each cell type --------
    for i, ct in enumerate(ctypes):
        arr = p_map[..., i]
        out_png = os.path.join(out_intensity, f"masked_{safe_name(ct)}.png")
        save_masked_intensity(arr, mask, out_png, title=f"{sample} – Masked {ct}", p_lo=p_lo, p_hi=p_hi)

    # -------- Predicted cell-type argmax map --------
    argmax_png = os.path.join(out_sample, f"{sample}_predicted_celltype_map.png")
    save_argmax_celltype_map(p_map, ctypes, mask, argmax_png, base_palette="tab20")

    # -------- Minimal text summary --------
    summary_txt = os.path.join(out_sample, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(
            f"Sample: {sample}\n"
            f"z_map: {z_map.shape}, p_map: {p_map.shape}, K={K}, D={D}\n"
            f"Mask: {mask.shape} (True = tissue)\n"
            f"Outputs:\n"
            f" - Cluster map: {cluster_map_png}\n"
            f" - Per-cluster highlights: {'enabled' if save_highlights else 'disabled'}\n"
            f" - Cell-type intensity dir: {out_intensity}\n"
            f" - Argmax cell-type map: {argmax_png}\n"
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process one sample and export plots.")
    p.add_argument("--data-root", required=True, help="Root folder containing sample subfolders.")
    p.add_argument("--sample", required=True, help="Sample folder name (e.g., P8_LUAD).")
    p.add_argument("--out-root", required=True, help="Where to write outputs: <out-root>/<SAMPLE>/...")

    # Provide either --pkl-path OR (--pkl-name + --data-root)
    p.add_argument("--pkl-path", default=None, help="Full path to the prediction pickle.")
    p.add_argument("--pkl-name", default=None, help="Filename of the pickle inside the sample folder.")

    p.add_argument("--data-root-ori", default=None, help="Root for original assets (mask). Defaults to --data-root.")

    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--pca-comp", type=int, default=100)
    p.add_argument("--random-seed", type=int, default=0)

    p.add_argument("--save-highlights", action="store_true", help="Save per-cluster highlight PNGs.")
    p.add_argument("--p-lo", type=float, default=5.0, help="Lower percentile for intensity maps.")
    p.add_argument("--p-hi", type=float, default=95.0, help="Upper percentile for intensity maps.")
    return p.parse_args()

def main():
    args = parse_args()
    run_one_sample(
        data_root=args.data_root,
        sample=args.sample,
        out_root=args.out_root,
        pkl_name=args.pkl_name,
        pkl_path=args.pkl_path,
        data_root_ori=args.data_root_ori,
        n_clusters=args.n_clusters,
        pca_comp=args.pca_comp,
        random_seed=args.random_seed,
        save_highlights=args.save_highlights,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
    )

if __name__ == "__main__":
    main()
