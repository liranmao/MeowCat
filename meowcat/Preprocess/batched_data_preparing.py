########################
# Visium
########################

import sys
sys.path.append('/home/liranmao/06_he_anno/code/1_main_lung_cancer/')

# --- edit me (or pass --config path/to/config.yaml to override) ---
PREFIX = "/project/KidneyHE/data_lung/7_new_sc_data/"   # contains subfolders like P001/, P002/, ...
OUT_DIR = f"{PREFIX}/batches_visium_subset_all_sample_new_p21_out"  # where we'll write batch_000.pkl

# coreset knobs
KEEP_FRAC = 0.25        # keep 25% of training spots overall
STRATEGY  = "stratified"   # "kcenter" or "stratified"
SEED      = 0

# optional: include/exclude certain sample folders by *folder name*
INCLUDE_ONLY = None      # e.g., {"P001","P005"} or None
EXCLUDE_SET  = {'P21_LUAD'}     # e.g., {"P_bad1","P_bad2"}

# ---- config override (do not edit below this line) ----
import argparse as _ap, yaml as _yaml
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument('--config', default=None)
_known, _ = _p.parse_known_args()
if _known.config:
    with open(_known.config) as _f:
        _cfg = _yaml.safe_load(_f) or {}
    _proj    = _cfg.get('project', {})
    _batches = _cfg.get('batches', {})
    if _proj.get('data_root'):   PREFIX       = _proj['data_root']
    if _batches.get('out_dir'):  OUT_DIR      = _batches['out_dir']
    if _batches.get('keep_frac') is not None: KEEP_FRAC    = _batches['keep_frac']
    if _batches.get('strategy'): STRATEGY     = _batches['strategy']
    if _batches.get('seed') is not None:      SEED         = _batches['seed']
    if _batches.get('include_only') is not None: INCLUDE_ONLY = set(_batches['include_only']) if _batches['include_only'] else None
    if _batches.get('exclude_set'): EXCLUDE_SET  = set(_batches['exclude_set'])
# -------------------------------------------------------

# ----- imports -----
import os, gc, math, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# you already have these in your codebase:
from utils import read_lines, read_string, save_pickle, load_pickle
from impute_by_basic import get_gene_counts, get_locs
from image import get_disk_mask
# If they are not in sys.path in your notebook, adjust sys.path to import them.


# --- tiny helpers that mirror your script's logic ---

def read_string(path):
    with open(path, "r") as f:
        return f.read().strip()

def _read_radius(sample_dir):
    # mirror your code: integer radius stored in radius.txt, divided by 16
    try:
        return int(read_string(os.path.join(sample_dir, 'radius.txt'))) // 16
    except Exception:
        return None

def auto_domain_map(sample_dirs):
    """Auto one domain per WSI folder name (like your _auto_domain_map_from_samples)."""
    names = [os.path.basename(os.path.normpath(d)) for d in sample_dirs]
    uniq = sorted(set(names))
    name_to_id = {n: i for i, n in enumerate(uniq)}
    return name_to_id, uniq

def load_domain_map_tsv(tsv_path):
    """Optional TSV: sample_name \t domain_string  -> ids 0..D-1"""
    name_to_domstr = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            k, v = line.split('\t')
            name_to_domstr[k] = v
    dom_names = sorted(set(name_to_domstr.values()))
    dom_to_id = {d:i for i,d in enumerate(dom_names)}
    return {k: dom_to_id[v] for k,v in name_to_domstr.items()}, dom_names

def list_sample_dirs(prefix, include_only=None, exclude_set=None):
    all_subdirs = [d for d in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, d))]
    if include_only is None:
        base = [d for d in all_subdirs if d.startswith("P")]
    else:
        base = sorted(list(set(all_subdirs) & set(include_only)))
    if exclude_set:
        base = [d for d in base if d not in exclude_set]
    sample_dirs = [os.path.join(prefix, d) + os.sep for d in sorted(base)]
    return sample_dirs

def get_cnts_data(prefix):
    # your original function, inlined here for notebook convenience
    from utils import read_lines, load_pickle
    from impute_by_basic import get_gene_counts, get_locs
    gene_names = read_lines(f'{prefix}anno-names.txt')   # cell-type names
    cnts = get_gene_counts(prefix)                       # DF [N_spots, K]
    cnts = cnts[gene_names]                              # keep correct col order
    # embs = load_pickle(f'{prefix}embeddings-hist.pickle')# [H,W,C]
    # locs = get_locs(prefix, target_shape=embs.shape[:2]) # [N_spots, 2]
    return cnts


def get_data(prefix):
    # your original function, inlined here for notebook convenience
    from utils import read_lines, load_pickle
    from impute_by_basic import get_gene_counts, get_locs
    gene_names = read_lines(f'{prefix}anno-names.txt')   # cell-type names
    cnts = get_gene_counts(prefix)                       # DF [N_spots, K]
    cnts = cnts[gene_names]                              # keep correct col order
    embs = load_pickle(f'{prefix}embeddings-hist.pickle')# [H,W,C]
    locs = get_locs(prefix, target_shape=embs.shape[:2]) # [N_spots, 2]
    return embs, cnts, locs

# 
def get_patches_tokens(img, locs, mask):
    """
    Your per-spot tokenization used in the training script.
    img: [H,W,C], mask: [2r,2r] boolean disk
    Returns [N_spots, T, C]
    """
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)
    x_list = []
    for s in locs:
        patch = img[s[0]+r[0][0]:s[0]+r[0][1], s[1]+r[1][0]:s[1]+r[1][1]]  # [2r,2r,C]
        x = patch[mask]  # [T,C]
        x_list.append(x)
    return np.stack(x_list)


# Optional: if you have a domain map TSV, set its path here; else auto-per-WSI
DOMAIN_MAP_TSV = None  # e.g., f"{PREFIX}/domain_map.tsv"

sample_dirs = list_sample_dirs(PREFIX, include_only=INCLUDE_ONLY, exclude_set=EXCLUDE_SET)
assert len(sample_dirs) > 0, "No sample folders found."

if DOMAIN_MAP_TSV and os.path.exists(DOMAIN_MAP_TSV):
    name_to_domain, domain_names = load_domain_map_tsv(DOMAIN_MAP_TSV)
else:
    name_to_domain, domain_names = auto_domain_map(sample_dirs)

# pass 1: read labels (no tokenization), collect global index mapping
Y_rows   = []
D_rows   = []
S_ptrs   = []   # (sample_idx, local_row_index) for each global row
N_per_WSI= []

for s_idx, sdir in enumerate(sample_dirs):
    try:
        cnts = get_cnts_data(sdir)    # embs is not used here; we avoid heavy ops
        
        n = len(cnts)
        N_per_WSI.append(n)
        y = cnts.to_numpy(dtype=np.float32, copy=True)   # [n, K]
        Y_rows.append(y)
        dom_id = name_to_domain[os.path.basename(os.path.normpath(sdir))]
        D_rows.append(np.full((n,), dom_id, dtype=np.int64))
        S_ptrs.extend([(s_idx, i) for i in range(n)])
        del cnts, y
    except Exception as e:
        print(f"[scan] skip {os.path.basename(os.path.normpath(sdir))}: {e}")

Y_all = np.concatenate(Y_rows, axis=0) if Y_rows else np.empty((0,0), dtype=np.float32)  # [N,K]
D_all = np.concatenate(D_rows, axis=0) if D_rows else np.empty((0,), dtype=np.int64)     # [N]

N_total, K = Y_all.shape if Y_all.size else (0, 0)
print(f"Scanned {len(sample_dirs)} samples | N_total={N_total}, K={K}, domains={len(domain_names)}")


def select_coreset_labels(Y, D, keep_frac=0.25, strategy="kcenter", n_entropy_bins=3, seed=0):
    """
    Y: [N,K] soft labels, D: [N] domain ids (>=0), returns global indices kept.
    """
    rng = np.random.default_rng(seed)
    N = Y.shape[0]
    Kkeep_total = max(1, int(np.ceil(keep_frac * N)))

    def _kcenter_idx(Ym, K):
        n = Ym.shape[0]
        if n <= K: return np.arange(n)
        start = int(rng.integers(0, n))
        centers = [start]
        d2 = np.sum((Ym - Ym[start])**2, axis=1)
        for _ in range(1, K):
            i = int(np.argmax(d2))
            centers.append(i)
            d2 = np.minimum(d2, np.sum((Ym - Ym[i])**2, axis=1))
        return np.array(centers, dtype=int)

    def _stratified_idx(Ym, K, n_bins):
        p = np.clip(Ym, 1e-8, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        H = -(p * np.log(p)).sum(axis=1)
        cls = np.argmax(Ym, axis=1)
        edges = np.quantile(H, np.linspace(0, 1, n_bins+1))
        edges[0] -= 1e-6; edges[-1] += 1e-6
        strata = defaultdict(list)
        for i in range(Ym.shape[0]):
            b = int(np.clip(np.searchsorted(edges, H[i], side='right') - 1, 0, n_bins-1))
            strata[(int(cls[i]), b)].append(i)
        sizes = {k: len(v) for k, v in strata.items()}
        total = sum(sizes.values())
        alloc = {k: max(1, int(round(K * sizes[k] / max(1, total)))) for k in strata.keys()}
        drift = K - sum(alloc.values())
        keys = list(strata.keys())
        while drift != 0:
            k = keys[rng.integers(0, len(keys))]
            if drift > 0:
                alloc[k] += 1; drift -= 1
            elif alloc[k] > 1:
                alloc[k] -= 1; drift += 1
        sel = []
        for k, idxs in strata.items():
            m = min(alloc[k], len(idxs))
            if m > 0:
                sel.extend(rng.choice(idxs, size=m, replace=False))
        return np.array(sel, dtype=int)

    kept = []
    domains = sorted(set(int(d) for d in D if d >= 0))
    counts  = {d: int((D == d).sum()) for d in domains}
    for d in domains:
        idx_d = np.where(D == d)[0]
        Yd = Y[idx_d]
        Kd = max(1, int(round(Kkeep_total * counts[d] / max(1, N))))
        if strategy == "kcenter":
            sel_local = _kcenter_idx(Yd, Kd)
        else:
            sel_local = _stratified_idx(Yd, Kd, n_entropy_bins)
        kept.append(idx_d[sel_local])

    kept_idx = np.concatenate(kept)
    kept_idx.sort()
    return kept_idx

kept_idx = select_coreset_labels(Y_all, D_all, keep_frac=KEEP_FRAC, strategy=STRATEGY, seed=SEED)
print(f"[coreset] kept {len(kept_idx)} / {len(Y_all)} ({len(kept_idx)/max(1,len(Y_all)):.1%})")



os.makedirs(OUT_DIR, exist_ok=True)

# map kept global indices -> per-sample local indices
kept_by_sample = defaultdict(list)
for gi in kept_idx:
    s_idx, local_i = S_ptrs[gi]
    kept_by_sample[s_idx].append(local_i)

# build and save
X_list, Y_list, D_list = [], [], []

for s_idx, local_idxs in kept_by_sample.items():
    sdir = sample_dirs[s_idx]
    name = os.path.basename(os.path.normpath(sdir))
    try:
        embs, cnts, locs = get_data(sdir)          # embs: [H,W,C], cnts: DF [n,K], locs: [n,2]
        r = _read_radius(sdir)
        assert r is not None, f"missing/invalid radius in {name}"
        from image import get_disk_mask
        mask = get_disk_mask(r)                    # [2r,2r] boolean
        local_idxs_sorted = np.array(sorted(local_idxs), dtype=int)

        # subset: labels & locs
        y_sel = cnts.iloc[local_idxs_sorted].to_numpy(dtype=np.float32, copy=True)   # [m,K]
        locs_sel = locs[local_idxs_sorted]                                           # [m,2]; numpy array

        # tokenize ONLY the selected spots
        x_sel = get_patches_tokens(embs, locs_sel, mask)                             # [m,T,C]
        dom_id = name_to_domain[name]
        d_sel = np.full((x_sel.shape[0],), dom_id, dtype=np.int64)

        # sanity: finite tokens
        ok = np.isfinite(x_sel).all(axis=-1).all(axis=-1)
        x_sel = x_sel[ok]
        y_sel = y_sel[ok]
        d_sel = d_sel[ok]

        if len(x_sel) > 0:
            X_list.append(x_sel.astype(np.float16, copy=False))
            Y_list.append(y_sel.astype(np.float32, copy=False))
            D_list.append(d_sel.astype(np.int64,  copy=False))
            print(f"[save] {name}: +{len(x_sel)}")
        else:
            print(f"[save] {name}: no valid spots after filtering")
        del embs, cnts, locs, x_sel, y_sel, d_sel, mask, locs_sel
        gc.collect()
    except Exception as e:
        print(f"[save] skip {name}: {e}")

if not X_list:
    raise RuntimeError("Nothing to save — coreset empty or tokenization failed.")

X = np.concatenate(X_list, axis=0)   # [N_kept, T, C], float16
Y = np.concatenate(Y_list, axis=0)   # [N_kept, K], float32
D = np.concatenate(D_list, axis=0)   # [N_kept],     int64

batch_path = os.path.join(OUT_DIR, "batch_000.pkl")
with open(batch_path, "wb") as f:
    pickle.dump((X, Y, D), f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved: {batch_path}")
print(f"Shapes: X{X.shape} Y{Y.shape} D{D.shape} | T={X.shape[1]} C={X.shape[2]} K={Y.shape[1]}")



import pickle
import numpy as np
import os

# paths
pkl_path = "/project/KidneyHE/data_lung/7_new_sc_data/batches_visium_subset_all_sample_new_p21_out/batch_000.pkl"
out_dir = "/project/KidneyHE/data_lung/7_new_sc_data/batches_visium_subset_per_sample_new"

os.makedirs(out_dir, exist_ok=True)

# load pkl
with open(pkl_path, "rb") as f:
    X, Y, D = pickle.load(f)

print(f"Loaded: X{X.shape} Y{Y.shape} D{D.shape}")
print(f"Unique domains: {np.unique(D)}")

# split by domain and save
unique_domains = np.unique(D)
for dom_id in unique_domains:
    mask = D == dom_id
    x_sel = X[mask]  # [n, T, C]
    y_sel = Y[mask]  # [n, K]
    d_sel = D[mask]  # [n]

    # save as 3 .npy files per sample
    np.save(os.path.join(out_dir, f"batch_vis_{dom_id:03d}_x.npy"), x_sel.astype(np.float16))
    np.save(os.path.join(out_dir, f"batch_vis_{dom_id:03d}_y.npy"), y_sel.astype(np.float32))
    np.save(os.path.join(out_dir, f"batch_vis_{dom_id:03d}_d.npy"), d_sel.astype(np.int64))

    print(f"[save] domain {dom_id}: {x_sel.shape[0]} spots")

print(f"\nDone. Saved {len(unique_domains)} samples to {out_dir}")






########################
# Xenium
########################
# start prepare after reformat
# first get adata_cellbin with image feature, then for each sample, save 3 npy file.

from __future__ import annotations
import os, json, logging, pickle, re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image


import matplotlib.pyplot as plt


from sklearn.neighbors import KDTree
from sklearn.metrics import (
classification_report, confusion_matrix, accuracy_score,
balanced_accuracy_score, top_k_accuracy_score, roc_curve, auc,
)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

Image.MAX_IMAGE_PIXELS = None
%matplotlib inline



#################
for sample in ['P24_LUAD_Xenium', 'P19_LUAD_Xenium']:
    print(f'start {sample}')
    pt_data = '/project/CATCH/dataset/fuduanData/Xenium_data/'+ sample + '/adata_cellbin_HistoSweep.h5ad'
    data_path = '/project/KidneyHE/data_lung/00_luad_xenium/'+ sample + '/'+ sample + '/'
    sample_folder = '/project/KidneyHE/data_lung/00_luad_xenium/binned/new_sc_model/'+ sample + '/'


    adata_cellbin = sc.read(pt_data)
    adata_xenium = sc.read_10x_h5(data_path +'cell_feature_matrix.h5')
    cells_parquet = pd.read_parquet(data_path+'cells.parquet')
    micro_loc = cells_parquet[["x_centroid", "y_centroid"]].to_numpy()

    pixel_size_raw = 0.2125 ### should be, can check experiment.xenium data to confirm
    pxl_loc = micro_loc/pixel_size_raw

    adata_xenium.obsm['spatial'] = pxl_loc


    adata_histology = sc.read('/project/KidneyHE/data_lung/00_luad_xenium/binned/'+ sample + '/single_super_emb.h5ad')

    ann_path = '/project/KidneyHE/data_lung/00_luad_xenium/xenium_cell_type_anno/'+ sample + '.csv'
    ann = pd.read_csv(ann_path, sep=None, engine="python") 


    ################# align adata_histology with adata_cellbin
    # combine image feature with bin gene expression
    adata_cellbin.obsm["gene_expression"] = adata_cellbin.X.copy()

    # 1) make sure obs are aligned (you said they are the same)
    assert (adata_cellbin.obs_names == adata_histology.obs_names).all(), "obs mismatch between bins & histology"

    # 2) attach histology features as an obsm block
    adata_cellbin.obsm["histology_2048"] = (
        adata_histology.X.A if hasattr(adata_histology.X, "A") else adata_histology.X
    )
    # 3) keep the feature names somewhere handy
    adata_cellbin.uns["histology_2048_var_names"] = np.array(adata_histology.var_names)



    ############### process adata_xenium
    cell_type_mapping = {
        # --- B & Plasma ---
        'B': 'B',
        'Plasma': 'Plasma',

        # --- Myeloid (Macs, Monos, DCs, Neutrophils, Mast) ---
        'Myeloid_C7_Langhans': 'Myeloid',
        'Myeloid_C17_Mac_CCL5': 'Myeloid',
        'Myeloid_C1_Mac_alveolar': 'Myeloid',
        'Myeloid_C9_Mac_MMP9': 'Myeloid',
        'Myeloid_C15_Mac_IL1B_NLRP3': 'Myeloid',
        'Myeloid_C8_Mac_prolif': 'Myeloid',
        'Myeloid_C6_Mac_SPP1_MMP12': 'Myeloid',
        'Myeloid_C5_Mac_CXCL9_CXCL10': 'Myeloid',
        'Myeloid_C16_Mac_CCL4L2_CCL2': 'Myeloid',
        'Myeloid_C0_Mac_SLC40A1': 'Myeloid',
        'Myeloid_C2_Mac_MARCO': 'Myeloid',
        'Myeloid_C3_Neutro': 'Myeloid',
        'Myeloid_C4_cDC2': 'Myeloid',
        'Myeloid_C10_cDC1': 'Myeloid',
        'Myeloid_C12_DC_LAMP3': 'Myeloid',
        'Myeloid_C11_pDC': 'Myeloid',
        'Myeloid_C18_Mac_str': 'Myeloid',
        'Myeloid_C13_CCL19': 'Myeloid',
        'Mast': 'Myeloid',  # Mapped to Myeloid as it is the closest lineage fit

        # --- Stromal (Endothelial, Fibroblasts, Pericytes, SMC) ---
        'Stromal_C1_myCAF': 'Stromal',
        'Stromal_C0_Vas_endo_capillary': 'Stromal',
        'Stromal_C2_Fib_alveolar': 'Stromal',
        'Stromal_C5_Lym_endo': 'Stromal',
        'Stromal_C4_pericyte': 'Stromal',
        'Stromal_C7_Vas_endo_venous': 'Stromal',
        'Stromal_C6_Fib_PI16': 'Stromal',
        'Stromal_C3_SMC': 'Stromal',
        'Stromal_C8_iCAF_CCL2': 'Stromal',

        # --- Non-Tumor Epithelial ---
        'Club': 'NonTumor_Epi',
        'Basal': 'NonTumor_Epi',
        'Ciliated': 'NonTumor_Epi',
        'Goblet': 'NonTumor_Epi',
        'Ciliated_prolif': 'NonTumor_Epi',
        'AT1': 'NonTumor_Epi',
        'AT2_inflam': 'NonTumor_Epi',
        'AT2': 'NonTumor_Epi',

        # --- Tumor Epithelial ---
        'NE': 'Tumor_Epi',
        'KAC': 'Tumor_Epi',
        'KAC_prolif': 'Tumor_Epi',
        'Tumor_epi_LUAD': 'Tumor_Epi',

        # --- T Cells ---
        'CD4_Tmem_IL7R': 'T',
        'T_str_JUN': 'T',
        'CD8_Tem_GZMK': 'T',
        'CD8_Teff_CCL4L2': 'T',
        'CD8_Tex': 'T',
        'CD4_T_CD69_ICOS': 'T',
        'CD4_Treg_FOXP3': 'T',
        'CD4_Tmem_CD44': 'T',
        'T_prolif': 'T',
        'T_isg': 'T',
        'CD4_Tfh_CXCL13': 'T',

        # --- NK Cells ---
        'NK': 'NK',
    }

    adata_xenium.obs = adata_xenium.obs.join(ann.set_index("cell_id"), how="left")
    adata_xenium.obs['cell_type_major'] = adata_xenium.obs['cell_state'].map(cell_type_mapping)

    
    
    ########### make sure it is aligned

    # If you already have x_xy0, b_xy0 from earlier, reuse them.
    # Otherwise:
    x_xy0  = adata_xenium.obsm["spatial"].astype(np.float32)
    b_xy0  = adata_cellbin.obsm["transformed_pxl_loc_in_morphology"].astype(np.float32)
    # x_xy0 = x_xy - x_xy.min(axis=0, keepdims=True)
    # b_xy0 = b_xy - b_xy.min(axis=0, keepdims=True)

    # Downsample for visibility/speed
    MAX_CELLS = 300_000
    MAX_BINS  = 500_000
    rng = np.random.default_rng(0)
    xc_idx = rng.choice(x_xy0.shape[0], size=min(MAX_CELLS, x_xy0.shape[0]), replace=False)
    bn_idx = rng.choice(b_xy0.shape[0], size=min(MAX_BINS , b_xy0.shape[0]), replace=False)

    # Colors + sizes
    CELL_COLOR = "#1f77b4"  # blue
    BIN_COLOR  = "#ff7f0e"  # orange
    BIN_SIZE   = 1
    CELL_SIZE  = 1

    plt.figure(figsize=(9, 9))
    # bins (underlay)
    plt.scatter(
        b_xy0[bn_idx, 0], b_xy0[bn_idx, 1],
        c=BIN_COLOR, s=BIN_SIZE, alpha=0.25, linewidths=0, zorder=1, rasterized=True
    )
    # cells (overlay)
    plt.scatter(
        x_xy0[xc_idx, 0], x_xy0[xc_idx, 1],
        c=CELL_COLOR, s=CELL_SIZE, alpha=0.8, linewidths=0, zorder=2, rasterized=True
    )

    plt.gca().invert_yaxis()
    plt.axis("equal"); plt.axis("off")
    plt.title("Overlay: cells (blue) vs. bins (orange)")

    # legend
    legend = [
        Patch(facecolor=CELL_COLOR, edgecolor="black", label="Cells"),
        Patch(facecolor=BIN_COLOR,  edgecolor="black", label="Bins"),
    ]
    plt.legend(handles=legend, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, title="Layers")

    plt.tight_layout()
    plt.show()
    plt.savefig(sample_folder+"/alignment_visualization.png", dpi=300, bbox_inches="tight")

    
    adata_cellbin.write_h5ad(sample_folder+'Xenium_adata_cellbin_analysis_qv20.h5ad')
    adata_xenium.write_h5ad(sample_folder+'Xenium_adata_cell.h5ad')
    print(f'finished {sample}')


    
    
"""
Xenium Data Preparation for CDAN Training
==========================================
Prepares Xenium spatial transcriptomics data to match Visium batch format.
Outputs: batch_xen_00X_{x,y,d}.npy files + batch_sample_mapping.json
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
import gc

# =========================================================
# 1. CONFIGURATION
# =========================================================

SAMPLES = [
    'P11_LUAD_Xenium',
    'P17_LUAD_Xenium',  
    'P19_LUAD_Xenium',
    'P21_LUAD_Xenium',
    'P24_LUAD_Xenium',
]

# Base paths
DATA_BASE = '/project/KidneyHE/data_lung/00_luad_xenium/'
PROCESSED_BASE = '/project/KidneyHE/data_lung/00_luad_xenium/binned/new_sc_model/'
OUTPUT_DIR = "/project/KidneyHE/data_lung/7_new_sc_data/batches_xenium_subset_per_sample_new/"
VISIUM_SAMPLE_DIR = "/project/KidneyHE/data_lung/7_new_sc_data/P15_LUAD/"
VISIUM_OUTPUT_DIR = '/project/KidneyHE/data_lung/7_new_sc_data/batches_visium_subset_per_sample_new/'

# Processing parameters
RANDOM_SEED = 42

# Radius options (choose one):
# Option 1: Fixed radius in pixels
FIXED_RADIUS = 75.20  # Set to None to use dynamic radius

# Option 2: Dynamic radius based on bin spacing (only used if FIXED_RADIUS is None)
RADIUS_MULTIPLIER = 2.0

# Cell type mapping (fine -> coarse)
CELL_TYPE_MAPPING = {
    # B cells
    'B': 'B', 'Plasma': 'Plasma',
    # Myeloid
    'Myeloid_C7_Langhans': 'Myeloid', 'Myeloid_C17_Mac_CCL5': 'Myeloid',
    'Myeloid_C1_Mac_alveolar': 'Myeloid', 'Myeloid_C9_Mac_MMP9': 'Myeloid',
    'Myeloid_C15_Mac_IL1B_NLRP3': 'Myeloid', 'Myeloid_C8_Mac_prolif': 'Myeloid',
    'Myeloid_C6_Mac_SPP1_MMP12': 'Myeloid', 'Myeloid_C5_Mac_CXCL9_CXCL10': 'Myeloid',
    'Myeloid_C16_Mac_CCL4L2_CCL2': 'Myeloid', 'Myeloid_C0_Mac_SLC40A1': 'Myeloid',
    'Myeloid_C2_Mac_MARCO': 'Myeloid', 'Myeloid_C3_Neutro': 'Myeloid',
    'Myeloid_C4_cDC2': 'Myeloid', 'Myeloid_C10_cDC1': 'Myeloid',
    'Myeloid_C12_DC_LAMP3': 'Myeloid', 'Myeloid_C11_pDC': 'Myeloid',
    'Myeloid_C18_Mac_str': 'Myeloid', 'Myeloid_C13_CCL19': 'Myeloid',
    'Mast': 'Myeloid',
    # Stromal
    'Stromal_C1_myCAF': 'Stromal', 'Stromal_C0_Vas_endo_capillary': 'Stromal',
    'Stromal_C2_Fib_alveolar': 'Stromal', 'Stromal_C5_Lym_endo': 'Stromal',
    'Stromal_C4_pericyte': 'Stromal', 'Stromal_C7_Vas_endo_venous': 'Stromal',
    'Stromal_C6_Fib_PI16': 'Stromal', 'Stromal_C3_SMC': 'Stromal',
    'Stromal_C8_iCAF_CCL2': 'Stromal',
    # Non-tumor epithelial
    'Club': 'NonTumor_Epi', 'Basal': 'NonTumor_Epi',
    'Ciliated': 'NonTumor_Epi', 'Goblet': 'NonTumor_Epi',
    'Ciliated_prolif': 'NonTumor_Epi', 'AT1': 'NonTumor_Epi',
    'AT2_inflam': 'NonTumor_Epi', 'AT2': 'NonTumor_Epi',
    # Tumor epithelial
    'NE': 'Tumor_Epi', 'KAC': 'Tumor_Epi',
    'KAC_prolif': 'Tumor_Epi', 'Tumor_epi_LUAD': 'Tumor_Epi',
    # T cells
    'CD4_Tmem_IL7R': 'T', 'T_str_JUN': 'T',
    'CD8_Tem_GZMK': 'T', 'CD8_Teff_CCL4L2': 'T',
    'CD8_Tex': 'T', 'CD4_T_CD69_ICOS': 'T',
    'CD4_Treg_FOXP3': 'T', 'CD4_Tmem_CD44': 'T',
    'T_prolif': 'T', 'T_isg': 'T', 'CD4_Tfh_CXCL13': 'T',
    # NK
    'NK': 'NK',
}


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def get_sample_paths(sample_name):
    """Generate paths for a given sample."""
    return {
        'data_path': os.path.join(DATA_BASE, sample_name, sample_name),
        'processed_path': os.path.join(PROCESSED_BASE, sample_name),
    }


def load_visium_reference(visium_dir, output_dir):
    """Load Visium batch dimensions and class order."""
    # Get dimensions from existing Visium batch
    vis_files = sorted(glob.glob(os.path.join(output_dir, "batch_*_x.npy")))
    assert len(vis_files) > 0, "No Visium batches found in OUTPUT_DIR"
    
    sample_vis = np.load(vis_files[0], mmap_mode="r")
    batch_size, max_tokens, feature_dim = sample_vis.shape
    
    # Load class order
    anno_path = os.path.join(visium_dir, "anno-names.txt")
    with open(anno_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return {
        'batch_size': batch_size,
        'max_tokens': max_tokens,
        'feature_dim': feature_dim,
        'class_names': class_names,
        'class_to_idx': {c: i for i, c in enumerate(class_names)},
    }


def load_xenium_data(processed_path, min_counts=50, min_genes=20):
    """Load and filter Xenium cell and cellbin data."""
    adata_cell = sc.read(os.path.join(processed_path, 'Xenium_adata_cell.h5ad'))
    
    # Quality filtering
    sc.pp.filter_cells(adata_cell, min_counts=min_counts)
    sc.pp.filter_cells(adata_cell, min_genes=min_genes)
    
    # Load cellbin data (backed mode for memory efficiency)
    adata_cellbin = sc.read(
        os.path.join(processed_path, 'Xenium_adata_cellbin_analysis_qv20.h5ad'),
        backed='r'
    )
    
    return adata_cell, adata_cellbin


def create_label_matrix(adata_cell, class_to_idx, cell_type_mapping):
    """Create one-hot encoded label matrix aligned with Visium classes."""
    n_cells = adata_cell.n_obs
    n_classes = len(class_to_idx)
    
    # Map fine labels to coarse
    mapped_labels = adata_cell.obs['cell_state'].map(cell_type_mapping).fillna("Unknown").values
    
    # Create one-hot matrix
    Y = np.zeros((n_cells, n_classes), dtype=np.float32)
    valid_mask = np.zeros(n_cells, dtype=bool)
    
    for i, label in enumerate(mapped_labels):
        if label in class_to_idx:
            Y[i, class_to_idx[label]] = 1.0
            valid_mask[i] = True
    
    return Y, valid_mask, mapped_labels


def compute_bin_to_cell_mapping(adata_cell, adata_cellbin, fixed_radius=None, radius_multiplier=2.0):
    """Map bins to cells using radius-based search from cell centroids.
    
    Args:
        adata_cell: AnnData object with cell data
        adata_cellbin: AnnData object with cellbin data
        fixed_radius: If provided, use this fixed radius (in pixels). Otherwise compute dynamically.
        radius_multiplier: Multiplier for bin spacing (only used if fixed_radius is None)
    """
    # Get coordinates
    bin_coords = adata_cellbin.obsm["transformed_pxl_loc_in_morphology"].astype(np.float32)
    cell_coords = adata_cell.obsm["spatial"].astype(np.float32)
    
    # Build KDTree and estimate bin spacing
    tree = KDTree(bin_coords)
    distances, _ = tree.query(bin_coords, k=2)
    px_per_bin = float(np.median(distances[:, 1]))
    
    # Determine radius
    if fixed_radius is not None:
        radius = fixed_radius
        print(f"  Using fixed radius: {radius:.2f} px")
    else:
        radius = px_per_bin * radius_multiplier
        print(f"  Using dynamic radius: {radius:.2f} px ({radius_multiplier}x bin spacing)")
    
    # Query bins within radius of each cell
    bin_indices_per_cell = tree.query_radius(cell_coords, r=radius)
    
    # Build mapping dataframe
    bin_idx_list, cell_idx_list = [], []
    for cell_idx, bin_idxs in enumerate(bin_indices_per_cell):
        if len(bin_idxs) > 0:
            bin_idx_list.extend(bin_idxs)
            cell_idx_list.extend([cell_idx] * len(bin_idxs))
    
    df_pairs = pd.DataFrame({
        'bin_idx': bin_idx_list,
        'cell_idx': cell_idx_list
    }).drop_duplicates().sort_values('cell_idx')
    
    # Compute stats
    stats = {
        'px_per_bin': px_per_bin,
        'radius': radius,
        'radius_mode': 'fixed' if fixed_radius is not None else 'dynamic',
        'cells_with_bins': df_pairs['cell_idx'].nunique(),
        'total_cells': len(cell_coords),
        'bins_per_cell_median': df_pairs.groupby('cell_idx').size().median(),
        'bins_per_cell_mean': df_pairs.groupby('cell_idx').size().mean(),
    }
    
    return df_pairs, stats


def stratified_sample_cells(df_pairs, valid_mask, labels, target_size, random_seed):
    """Select cells using stratified sampling to preserve class distribution."""
    # Get candidates: valid label + has bins
    cells_with_bins = df_pairs['cell_idx'].unique()
    candidates = np.intersect1d(cells_with_bins, np.where(valid_mask)[0])
    candidate_labels = labels[candidates]
    
    if len(candidates) <= target_size:
        print(f"  Note: Available cells ({len(candidates)}) <= target ({target_size}). Using all.")
        return candidates
    
    try:
        selected, _ = train_test_split(
            candidates,
            train_size=target_size,
            stratify=candidate_labels,
            random_state=random_seed
        )
        print(f"  Stratified sampling: {len(selected)} cells selected.")
    except ValueError as e:
        print(f"  Warning: Stratification failed ({e}). Using random sampling.")
        np.random.seed(random_seed)
        selected = np.random.choice(candidates, target_size, replace=False)
    
    return selected

def get_max_visium_domain_id(visium_output_dir):
    """Find the largest domain ID used in existing Visium batches."""
    d_files = sorted(glob.glob(os.path.join(visium_output_dir, "batch_*_d.npy")))
    
    if not d_files:
        print("  Warning: No Visium domain files found. Starting from 0.")
        return -1
    
    max_domain = -1
    for d_file in d_files:
        d = np.load(d_file)
        if d.size > 0:
            max_domain = max(max_domain, int(d.max()))
        del d
    
    print(f"  Max Visium domain ID: {max_domain}")
    return max_domain


def extract_features_for_cells(cell_indices, df_pairs, adata_cellbin, max_tokens, random_seed):
    """Extract and pad/truncate features for selected cells."""
    grouped = df_pairs.groupby('cell_idx')
    np.random.seed(random_seed)
    
    X_list = []
    for cell_idx in cell_indices:
        bin_idxs = grouped.get_group(cell_idx)['bin_idx'].values
        
        # Load features
        if 'histology_2048' in adata_cellbin.obsm.keys():
            feats = adata_cellbin.obsm['histology_2048'][bin_idxs]
        else:
            feats = adata_cellbin.X[bin_idxs]
        
        if hasattr(feats, 'toarray'):
            feats = feats.toarray()
        
        # Pad or truncate to max_tokens
        k = feats.shape[0]
        if k < max_tokens:
            n_repeats = (max_tokens // k) + 1
            token_tensor = np.tile(feats, (n_repeats, 1))[:max_tokens]
        else:
            choice = np.random.choice(k, max_tokens, replace=False)
            token_tensor = feats[choice]
        
        X_list.append(token_tensor)
    
    return np.stack(X_list)


def save_batch(X, Y, domain_id, batch_name, output_dir):
    """Save batch arrays to disk."""
    np.save(os.path.join(output_dir, f"{batch_name}_x.npy"), X.astype(np.float16))
    np.save(os.path.join(output_dir, f"{batch_name}_y.npy"), Y.astype(np.float32))
    np.save(os.path.join(output_dir, f"{batch_name}_d.npy"), 
            np.full(len(X), domain_id, dtype=np.int64))


# =========================================================
# 3. MAIN PROCESSING
# =========================================================

def process_sample(sample_name, batch_idx, ref_params, domain_id):
    """Process a single Xenium sample and return batch info."""
    print(f"\n{'='*60}")
    print(f"Processing: {sample_name} (batch_xen_{batch_idx:03d}, domain={domain_id})")
    print(f"{'='*60}")
    
    paths = get_sample_paths(sample_name)
    
    # Load data
    print("Loading Xenium data...")
    adata_cell, adata_cellbin = load_xenium_data(paths['processed_path'])
    print(f"  Cells after QC: {adata_cell.n_obs}")
    
    # Create labels
    print("Creating label matrix...")
    Y_all, valid_mask, labels = create_label_matrix(
        adata_cell, ref_params['class_to_idx'], CELL_TYPE_MAPPING
    )
    print(f"  Valid labeled cells: {valid_mask.sum()} / {adata_cell.n_obs}")
    
    # Map bins to cells
    print("Mapping bins to cells...")
    df_pairs, mapping_stats = compute_bin_to_cell_mapping(
        adata_cell, adata_cellbin, 
        fixed_radius=FIXED_RADIUS, 
        radius_multiplier=RADIUS_MULTIPLIER
    )
    print(f"  Bin spacing: {mapping_stats['px_per_bin']:.2f} px")
    print(f"  Search radius: {mapping_stats['radius']:.2f} px")
    print(f"  Cells with bins: {mapping_stats['cells_with_bins']} / {mapping_stats['total_cells']}")
    print(f"  Bins per cell: median={mapping_stats['bins_per_cell_median']:.0f}, "
          f"mean={mapping_stats['bins_per_cell_mean']:.1f}")
    
    # Stratified sampling
    print(f"Selecting cells (target: {ref_params['batch_size']})...")
    selected_cells = stratified_sample_cells(
        df_pairs, valid_mask, labels, ref_params['batch_size'], RANDOM_SEED
    )
    
    # Extract features
    print("Extracting features...")
    X = extract_features_for_cells(
        selected_cells, df_pairs, adata_cellbin, ref_params['max_tokens'], RANDOM_SEED
    )
    Y = Y_all[selected_cells]
    
    # Save batch
    batch_name = f"batch_xen_{batch_idx:03d}"
    print(f"Saving {batch_name} with domain_id={domain_id}...")
    save_batch(X, Y, domain_id, batch_name, OUTPUT_DIR)

    # Cleanup to free memory
    if hasattr(adata_cellbin, 'file'):
        adata_cellbin.file.close()
    del adata_cell, adata_cellbin, df_pairs, X, Y_all
    gc.collect()
    
    # Return batch info
    return {
        'batch_name': batch_name,
        'sample_name': sample_name,
        'n_cells': len(selected_cells),
        'domain_id': domain_id,  # <-- Add this
        'mapping_stats': mapping_stats,
    }


"""Main entry point."""
print("Xenium Batch Preparation for CDAN Training")
print("=" * 60)

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Visium reference
print("\nLoading Visium reference parameters...")
ref_params = load_visium_reference(VISIUM_SAMPLE_DIR, VISIUM_OUTPUT_DIR)
ref_params['batch_size'] = 14847
print(f"  Batch size: {ref_params['batch_size']}")
print(f"  Max tokens (T): {ref_params['max_tokens']}")
print(f"  Feature dim (C): {ref_params['feature_dim']}")
print(f"  Classes ({len(ref_params['class_names'])}): {ref_params['class_names']}")

# Get starting domain ID for Xenium (max Visium domain + 1)
print("\nDetermining Xenium domain IDs...")
max_visium_domain = get_max_visium_domain_id(VISIUM_OUTPUT_DIR)
xenium_domain_start = max_visium_domain + 1
print(f"  Xenium domain IDs will start from: {xenium_domain_start}")

# Print radius configuration
if FIXED_RADIUS is not None:
    print(f"\nRadius mode: FIXED = {FIXED_RADIUS} px")
else:
    print(f"\nRadius mode: DYNAMIC = {RADIUS_MULTIPLIER}x bin spacing")

# Process all samples - each gets unique domain ID
batch_mapping = {}
for idx, sample in enumerate(SAMPLES):
    domain_id = xenium_domain_start + idx  # Unique per slide
    batch_info = process_sample(sample, idx, ref_params, domain_id)
    batch_mapping[batch_info['batch_name']] = {
        'sample_name': batch_info['sample_name'],
        'n_cells': batch_info['n_cells'],
        'domain_id': batch_info['domain_id'],  # <-- Add this
        'px_per_bin': batch_info['mapping_stats']['px_per_bin'],
        'radius': batch_info['mapping_stats']['radius'],
        'radius_mode': batch_info['mapping_stats']['radius_mode'],
    }

# Save batch-sample mapping
mapping_path = os.path.join(OUTPUT_DIR, "batch_sample_mapping.json")
with open(mapping_path, 'w') as f:
    json.dump(batch_mapping, f, indent=2)
print(f"\n{'='*60}")
print(f"Saved batch-sample mapping to: {mapping_path}")


print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for batch_name, info in batch_mapping.items():
    print(f"  {batch_name}: {info['sample_name']} "
          f"(n={info['n_cells']}, domain={info['domain_id']})")
print("\nProcessing complete!")

