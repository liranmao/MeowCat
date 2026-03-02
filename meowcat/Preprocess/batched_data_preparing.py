########################
# Visium / Xenium batch preparation
# Usage:
#   python batched_data_preparing.py --config config.yaml --mode visium
#   python batched_data_preparing.py --config config.yaml --mode xenium
########################

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── argument parsing ────────────────────────────────────────────────────────
import argparse as _ap, yaml as _yaml
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument('--config', default=None)
_p.add_argument('--mode', default='visium', choices=['visium', 'xenium'])
_known, _ = _p.parse_known_args()

# ── Visium defaults (overridden by config batches/project section) ──────────
PREFIX = "/project/KidneyHE/data_lung/7_new_sc_data/"
OUT_DIR = f"{PREFIX}/batches_visium_subset_all_sample_new_p21_out"
KEEP_FRAC = 0.25
STRATEGY  = "stratified"
SEED      = 0
SAMPLE_PATTERN = "P*"
INCLUDE_ONLY = None
EXCLUDE_SET  = {'P21_LUAD'}
DOMAIN_MAP_TSV = None

# ── Xenium defaults (overridden by config xenium section) ───────────────────
XEN_XENIUM_DATA_ROOT = "/path/to/xenium/raw"
XEN_CELLBIN_ROOT = "/path/to/xenium/cellbin"
XEN_PROCESSED_ROOT = "/path/to/xenium/processed"
XEN_HISTOLOGY_ROOT = "/path/to/xenium/histology"
XEN_ANNO_DIR = "/path/to/xenium_cell_type_anno"
XEN_SAMPLE_PATTERN = "P*_Xenium"
XEN_INCLUDE_ONLY = None
XEN_EXCLUDE_SET = set()
XEN_OUT_DIR = "/path/to/xenium/batches"
XEN_PIXEL_SIZE_RAW = 0.2125
XEN_FIXED_RADIUS = 75.20
XEN_ANNO_NAMES_PATH = "/path/to/anno-names.txt"
XEN_CELL_TYPE_MAPPING_JSON = None
XEN_VISIUM_BATCH_DIR = None
XEN_SEED = 42

# ── config override ─────────────────────────────────────────────────────────
if _known.config:
    with open(_known.config) as _f:
        _cfg = _yaml.safe_load(_f) or {}
    # Visium overrides
    _proj    = _cfg.get('project', {})
    _batches = _cfg.get('batches', {})
    if _proj.get('data_root'):       PREFIX         = _proj['data_root']
    if _proj.get('sample_pattern'):  SAMPLE_PATTERN = _proj['sample_pattern']
    if _batches.get('out_dir'):      OUT_DIR        = _batches['out_dir']
    if _batches.get('keep_frac') is not None: KEEP_FRAC    = _batches['keep_frac']
    if _batches.get('strategy'): STRATEGY     = _batches['strategy']
    if _batches.get('seed') is not None:      SEED         = _batches['seed']
    if _batches.get('include_only') is not None: INCLUDE_ONLY = set(_batches['include_only']) if _batches['include_only'] else None
    if _batches.get('exclude_set'): EXCLUDE_SET  = set(_batches['exclude_set'])
    if _batches.get('domain_map_tsv'): DOMAIN_MAP_TSV = _batches['domain_map_tsv']
    # Xenium overrides
    _xen = _cfg.get('xenium', {})
    if _xen.get('xenium_data_root'): XEN_XENIUM_DATA_ROOT = _xen['xenium_data_root']
    if _xen.get('cellbin_root'):     XEN_CELLBIN_ROOT     = _xen['cellbin_root']
    if _xen.get('processed_root'):   XEN_PROCESSED_ROOT   = _xen['processed_root']
    if _xen.get('histology_root'):   XEN_HISTOLOGY_ROOT   = _xen['histology_root']
    if _xen.get('xenium_anno_dir'):  XEN_ANNO_DIR         = _xen['xenium_anno_dir']
    if _xen.get('sample_pattern'):   XEN_SAMPLE_PATTERN   = _xen['sample_pattern']
    if _xen.get('include_only') is not None:
        XEN_INCLUDE_ONLY = set(_xen['include_only']) if _xen['include_only'] else None
    if _xen.get('exclude_set'):      XEN_EXCLUDE_SET      = set(_xen['exclude_set'])
    if _xen.get('out_dir'):          XEN_OUT_DIR          = _xen['out_dir']
    if _xen.get('pixel_size_raw') is not None: XEN_PIXEL_SIZE_RAW = _xen['pixel_size_raw']
    if _xen.get('fixed_radius') is not None:   XEN_FIXED_RADIUS   = _xen['fixed_radius']
    if _xen.get('anno_names_path'):  XEN_ANNO_NAMES_PATH  = _xen['anno_names_path']
    if _xen.get('cell_type_mapping_json'): XEN_CELL_TYPE_MAPPING_JSON = _xen['cell_type_mapping_json']
    if _xen.get('visium_batch_dir'): XEN_VISIUM_BATCH_DIR = _xen['visium_batch_dir']
    if _xen.get('seed') is not None: XEN_SEED             = _xen['seed']

# ── imports ─────────────────────────────────────────────────────────────────
import gc, math, pickle, json
import glob as _glob_mod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import read_lines, load_pickle, save_pickle
from impute_by_basic import get_gene_counts, get_locs
from image import get_disk_mask


# =========================================================================
# SHARED HELPER FUNCTIONS
# =========================================================================

def read_string(path):
    with open(path, "r") as f:
        return f.read().strip()

def _read_radius(sample_dir):
    try:
        return int(read_string(os.path.join(sample_dir, 'radius.txt'))) // 16
    except Exception:
        return None

def auto_domain_map(sample_dirs):
    """Auto one domain per WSI folder name."""
    names = [os.path.basename(os.path.normpath(d)) for d in sample_dirs]
    uniq = sorted(set(names))
    name_to_id = {n: i for i, n in enumerate(uniq)}
    return name_to_id, uniq

def load_domain_map_tsv(tsv_path):
    """Optional TSV: sample_name \\t domain_string  -> ids 0..D-1"""
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

def list_sample_dirs(prefix, include_only=None, exclude_set=None, sample_pattern=None):
    if include_only is not None:
        all_subdirs = [d for d in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, d))]
        base = sorted(list(set(all_subdirs) & set(include_only)))
    elif sample_pattern:
        matches = sorted(_glob_mod.glob(os.path.join(prefix, sample_pattern)))
        base = [os.path.basename(m) for m in matches if os.path.isdir(m)]
    else:
        base = sorted(d for d in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, d)))
    if exclude_set:
        base = [d for d in base if d not in exclude_set]
    sample_dirs = [os.path.join(prefix, d) + os.sep for d in sorted(base)]
    return sample_dirs

def get_cnts_data(prefix):
    gene_names = read_lines(f'{prefix}anno-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    return cnts

def get_data(prefix):
    gene_names = read_lines(f'{prefix}anno-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = load_pickle(f'{prefix}embeddings-hist.pickle')
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts, locs

def get_patches_tokens(img, locs, mask):
    """Per-spot tokenization: img [H,W,C], mask [2r,2r] boolean -> [N,T,C]"""
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)
    x_list = []
    for s in locs:
        patch = img[s[0]+r[0][0]:s[0]+r[0][1], s[1]+r[1][0]:s[1]+r[1][1]]
        x = patch[mask]
        x_list.append(x)
    return np.stack(x_list)


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


# =========================================================================
# XENIUM-SPECIFIC HELPER FUNCTIONS
# =========================================================================

def load_cell_type_mapping(json_path):
    """Load cell type mapping from JSON file."""
    if json_path is None:
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


def list_xenium_samples(root, sample_pattern=None, include_only=None, exclude_set=None):
    """Discover Xenium sample names from a directory."""
    if include_only is not None:
        samples = sorted(set(include_only))
    elif sample_pattern:
        matches = sorted(_glob_mod.glob(os.path.join(root, sample_pattern)))
        samples = [os.path.basename(m) for m in matches if os.path.isdir(m)]
    else:
        samples = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if exclude_set:
        samples = [s for s in samples if s not in exclude_set]
    return samples


def xen_load_visium_reference(anno_names_path, visium_batch_dir):
    """Load class order from anno-names.txt and batch dims from Visium batches."""
    with open(anno_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    result = {
        'class_names': class_names,
        'class_to_idx': {c: i for i, c in enumerate(class_names)},
    }

    if visium_batch_dir:
        vis_files = sorted(_glob_mod.glob(os.path.join(visium_batch_dir, "batch_*_x.npy")))
        if vis_files:
            sample_vis = np.load(vis_files[0], mmap_mode="r")
            result['batch_size'] = sample_vis.shape[0]
            result['max_tokens'] = sample_vis.shape[1]
            result['feature_dim'] = sample_vis.shape[2]

    return result


def xen_load_xenium_data(processed_path, min_counts=50, min_genes=20):
    """Load and filter Xenium cell and cellbin data."""
    import scanpy as sc
    adata_cell = sc.read(os.path.join(processed_path, 'Xenium_adata_cell.h5ad'))
    sc.pp.filter_cells(adata_cell, min_counts=min_counts)
    sc.pp.filter_cells(adata_cell, min_genes=min_genes)
    adata_cellbin = sc.read(
        os.path.join(processed_path, 'Xenium_adata_cellbin_analysis_qv20.h5ad'),
        backed='r'
    )
    return adata_cell, adata_cellbin


def xen_create_label_matrix(adata_cell, class_to_idx, cell_type_mapping):
    """Create one-hot encoded label matrix aligned with Visium classes."""
    n_cells = adata_cell.n_obs
    n_classes = len(class_to_idx)
    mapped_labels = adata_cell.obs['cell_state'].map(cell_type_mapping).fillna("Unknown").values
    Y = np.zeros((n_cells, n_classes), dtype=np.float32)
    valid_mask = np.zeros(n_cells, dtype=bool)
    for i, label in enumerate(mapped_labels):
        if label in class_to_idx:
            Y[i, class_to_idx[label]] = 1.0
            valid_mask[i] = True
    return Y, valid_mask, mapped_labels


def xen_compute_bin_to_cell_mapping(adata_cell, adata_cellbin, fixed_radius):
    """Map bins to cells using radius-based KDTree search."""
    from sklearn.neighbors import KDTree
    bin_coords = adata_cellbin.obsm["transformed_pxl_loc_in_morphology"].astype(np.float32)
    cell_coords = adata_cell.obsm["spatial"].astype(np.float32)
    tree = KDTree(bin_coords)
    distances, _ = tree.query(bin_coords, k=2)
    px_per_bin = float(np.median(distances[:, 1]))
    radius = fixed_radius
    print(f"  Using fixed radius: {radius:.2f} px")
    bin_indices_per_cell = tree.query_radius(cell_coords, r=radius)
    bin_idx_list, cell_idx_list = [], []
    for cell_idx, bin_idxs in enumerate(bin_indices_per_cell):
        if len(bin_idxs) > 0:
            bin_idx_list.extend(bin_idxs)
            cell_idx_list.extend([cell_idx] * len(bin_idxs))
    df_pairs = pd.DataFrame({
        'bin_idx': bin_idx_list,
        'cell_idx': cell_idx_list
    }).drop_duplicates().sort_values('cell_idx')
    stats = {
        'px_per_bin': px_per_bin,
        'radius': radius,
        'cells_with_bins': df_pairs['cell_idx'].nunique(),
        'total_cells': len(cell_coords),
        'bins_per_cell_median': df_pairs.groupby('cell_idx').size().median(),
        'bins_per_cell_mean': df_pairs.groupby('cell_idx').size().mean(),
    }
    return df_pairs, stats


def xen_stratified_sample_cells(df_pairs, valid_mask, labels, target_size, random_seed):
    """Select cells using stratified sampling."""
    from sklearn.model_selection import train_test_split
    cells_with_bins = df_pairs['cell_idx'].unique()
    candidates = np.intersect1d(cells_with_bins, np.where(valid_mask)[0])
    candidate_labels = labels[candidates]
    if len(candidates) <= target_size:
        print(f"  Note: Available cells ({len(candidates)}) <= target ({target_size}). Using all.")
        return candidates
    try:
        selected, _ = train_test_split(
            candidates, train_size=target_size,
            stratify=candidate_labels, random_state=random_seed
        )
        print(f"  Stratified sampling: {len(selected)} cells selected.")
    except ValueError as e:
        print(f"  Warning: Stratification failed ({e}). Using random sampling.")
        np.random.seed(random_seed)
        selected = np.random.choice(candidates, target_size, replace=False)
    return selected


def xen_get_max_visium_domain_id(visium_batch_dir):
    """Find the largest domain ID in existing Visium batches."""
    if not visium_batch_dir:
        return -1
    d_files = sorted(_glob_mod.glob(os.path.join(visium_batch_dir, "batch_*_d.npy")))
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


def xen_extract_features_for_cells(cell_indices, df_pairs, adata_cellbin, max_tokens, random_seed):
    """Extract and pad/truncate features for selected cells."""
    grouped = df_pairs.groupby('cell_idx')
    np.random.seed(random_seed)
    X_list = []
    for cell_idx in cell_indices:
        bin_idxs = grouped.get_group(cell_idx)['bin_idx'].values
        if 'histology_2048' in adata_cellbin.obsm.keys():
            feats = adata_cellbin.obsm['histology_2048'][bin_idxs]
        else:
            feats = adata_cellbin.X[bin_idxs]
        if hasattr(feats, 'toarray'):
            feats = feats.toarray()
        k = feats.shape[0]
        if k < max_tokens:
            n_repeats = (max_tokens // k) + 1
            token_tensor = np.tile(feats, (n_repeats, 1))[:max_tokens]
        else:
            choice = np.random.choice(k, max_tokens, replace=False)
            token_tensor = feats[choice]
        X_list.append(token_tensor)
    return np.stack(X_list)


def xen_save_batch(X, Y, domain_id, batch_name, output_dir):
    """Save batch arrays to disk."""
    np.save(os.path.join(output_dir, f"{batch_name}_x.npy"), X.astype(np.float16))
    np.save(os.path.join(output_dir, f"{batch_name}_y.npy"), Y.astype(np.float32))
    np.save(os.path.join(output_dir, f"{batch_name}_d.npy"),
            np.full(len(X), domain_id, dtype=np.int64))


# =========================================================================
# VISIUM BATCH PREPARATION
# =========================================================================

def _run_visium():
    sample_dirs = list_sample_dirs(
        PREFIX, include_only=INCLUDE_ONLY, exclude_set=EXCLUDE_SET,
        sample_pattern=SAMPLE_PATTERN
    )
    assert len(sample_dirs) > 0, "No sample folders found."

    if DOMAIN_MAP_TSV and os.path.exists(DOMAIN_MAP_TSV):
        name_to_domain, domain_names = load_domain_map_tsv(DOMAIN_MAP_TSV)
    else:
        name_to_domain, domain_names = auto_domain_map(sample_dirs)

    # pass 1: read labels (no tokenization), collect global index mapping
    Y_rows   = []
    D_rows   = []
    S_ptrs   = []
    N_per_WSI= []

    for s_idx, sdir in enumerate(sample_dirs):
        try:
            cnts = get_cnts_data(sdir)
            n = len(cnts)
            N_per_WSI.append(n)
            y = cnts.to_numpy(dtype=np.float32, copy=True)
            Y_rows.append(y)
            dom_id = name_to_domain[os.path.basename(os.path.normpath(sdir))]
            D_rows.append(np.full((n,), dom_id, dtype=np.int64))
            S_ptrs.extend([(s_idx, i) for i in range(n)])
            del cnts, y
        except Exception as e:
            print(f"[scan] skip {os.path.basename(os.path.normpath(sdir))}: {e}")

    Y_all = np.concatenate(Y_rows, axis=0) if Y_rows else np.empty((0,0), dtype=np.float32)
    D_all = np.concatenate(D_rows, axis=0) if D_rows else np.empty((0,), dtype=np.int64)
    N_total, K = Y_all.shape if Y_all.size else (0, 0)
    print(f"Scanned {len(sample_dirs)} samples | N_total={N_total}, K={K}, domains={len(domain_names)}")

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
            embs, cnts, locs = get_data(sdir)
            r = _read_radius(sdir)
            assert r is not None, f"missing/invalid radius in {name}"
            mask = get_disk_mask(r)
            local_idxs_sorted = np.array(sorted(local_idxs), dtype=int)

            y_sel = cnts.iloc[local_idxs_sorted].to_numpy(dtype=np.float32, copy=True)
            locs_sel = locs[local_idxs_sorted]
            x_sel = get_patches_tokens(embs, locs_sel, mask)
            dom_id = name_to_domain[name]
            d_sel = np.full((x_sel.shape[0],), dom_id, dtype=np.int64)

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

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    D = np.concatenate(D_list, axis=0)

    # Save bulk pickle
    batch_path = os.path.join(OUT_DIR, "batch_000.pkl")
    with open(batch_path, "wb") as f:
        pickle.dump((X, Y, D), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved: {batch_path}")
    print(f"Shapes: X{X.shape} Y{Y.shape} D{D.shape} | T={X.shape[1]} C={X.shape[2]} K={Y.shape[1]}")

    # Split by domain into per-sample .npy files
    unique_domains = np.unique(D)
    for dom_id in unique_domains:
        dmask = D == dom_id
        x_sel = X[dmask]
        y_sel = Y[dmask]
        d_sel = D[dmask]
        np.save(os.path.join(OUT_DIR, f"batch_vis_{dom_id:03d}_x.npy"), x_sel.astype(np.float16))
        np.save(os.path.join(OUT_DIR, f"batch_vis_{dom_id:03d}_y.npy"), y_sel.astype(np.float32))
        np.save(os.path.join(OUT_DIR, f"batch_vis_{dom_id:03d}_d.npy"), d_sel.astype(np.int64))
        print(f"[save] domain {dom_id}: {x_sel.shape[0]} spots")

    print(f"\nDone. Saved {len(unique_domains)} samples to {OUT_DIR}")


# =========================================================================
# XENIUM BATCH PREPARATION
# =========================================================================

def _run_xenium():
    import scanpy as sc
    from PIL import Image
    from matplotlib.patches import Patch
    Image.MAX_IMAGE_PIXELS = None

    # Load cell type mapping from JSON
    cell_type_mapping = load_cell_type_mapping(XEN_CELL_TYPE_MAPPING_JSON)
    if cell_type_mapping is None:
        raise ValueError(
            "xenium.cell_type_mapping_json must be set in config. "
            "See config/cell_type_mapping_lung.json for an example."
        )

    # Discover samples
    samples = list_xenium_samples(
        XEN_PROCESSED_ROOT,
        sample_pattern=XEN_SAMPLE_PATTERN,
        include_only=XEN_INCLUDE_ONLY,
        exclude_set=XEN_EXCLUDE_SET,
    )
    assert len(samples) > 0, "No Xenium sample folders found."
    print(f"Found {len(samples)} Xenium samples: {samples}")

    # ── Part A: Preliminary h5ad processing ────────────────────────────────
    print("\n" + "=" * 60)
    print("Part A: Xenium Preliminary Processing")
    print("=" * 60)

    for sample in samples:
        print(f'\nProcessing {sample} ...')
        pt_data = os.path.join(XEN_CELLBIN_ROOT, sample, 'adata_cellbin_HistoSweep.h5ad')
        data_path = os.path.join(XEN_XENIUM_DATA_ROOT, sample, sample)
        sample_folder = os.path.join(XEN_PROCESSED_ROOT, sample)

        # Check if already processed — skip to avoid redundant work
        cellbin_out = os.path.join(sample_folder, 'Xenium_adata_cellbin_analysis_qv20.h5ad')
        cell_out = os.path.join(sample_folder, 'Xenium_adata_cell.h5ad')
        if os.path.exists(cellbin_out) and os.path.exists(cell_out):
            print(f'  [skip] h5ad files already exist for {sample}')
            continue

        os.makedirs(sample_folder, exist_ok=True)

        adata_cellbin = sc.read(pt_data)
        adata_xenium = sc.read_10x_h5(os.path.join(data_path, 'cell_feature_matrix.h5'))
        cells_parquet = pd.read_parquet(os.path.join(data_path, 'cells.parquet'))
        micro_loc = cells_parquet[["x_centroid", "y_centroid"]].to_numpy()

        pxl_loc = micro_loc / XEN_PIXEL_SIZE_RAW
        adata_xenium.obsm['spatial'] = pxl_loc

        histology_path = os.path.join(XEN_HISTOLOGY_ROOT, sample, 'single_super_emb.h5ad')
        adata_histology = sc.read(histology_path)

        ann_path = os.path.join(XEN_ANNO_DIR, sample + '.csv')
        ann = pd.read_csv(ann_path, sep=None, engine="python")

        # Align histology with cellbin
        adata_cellbin.obsm["gene_expression"] = adata_cellbin.X.copy()
        assert (adata_cellbin.obs_names == adata_histology.obs_names).all(), \
            "obs mismatch between bins & histology"
        adata_cellbin.obsm["histology_2048"] = (
            adata_histology.X.A if hasattr(adata_histology.X, "A") else adata_histology.X
        )
        adata_cellbin.uns["histology_2048_var_names"] = np.array(adata_histology.var_names)

        # Map cell types
        adata_xenium.obs = adata_xenium.obs.join(ann.set_index("cell_id"), how="left")
        adata_xenium.obs['cell_type_major'] = adata_xenium.obs['cell_state'].map(cell_type_mapping)

        # Alignment visualization
        x_xy0  = adata_xenium.obsm["spatial"].astype(np.float32)
        b_xy0  = adata_cellbin.obsm["transformed_pxl_loc_in_morphology"].astype(np.float32)

        MAX_CELLS = 300_000
        MAX_BINS  = 500_000
        rng = np.random.default_rng(0)
        xc_idx = rng.choice(x_xy0.shape[0], size=min(MAX_CELLS, x_xy0.shape[0]), replace=False)
        bn_idx = rng.choice(b_xy0.shape[0], size=min(MAX_BINS , b_xy0.shape[0]), replace=False)

        CELL_COLOR = "#1f77b4"
        BIN_COLOR  = "#ff7f0e"
        plt.figure(figsize=(9, 9))
        plt.scatter(
            b_xy0[bn_idx, 0], b_xy0[bn_idx, 1],
            c=BIN_COLOR, s=1, alpha=0.25, linewidths=0, zorder=1, rasterized=True
        )
        plt.scatter(
            x_xy0[xc_idx, 0], x_xy0[xc_idx, 1],
            c=CELL_COLOR, s=1, alpha=0.8, linewidths=0, zorder=2, rasterized=True
        )
        plt.gca().invert_yaxis()
        plt.axis("equal"); plt.axis("off")
        plt.title("Overlay: cells (blue) vs. bins (orange)")
        legend = [
            Patch(facecolor=CELL_COLOR, edgecolor="black", label="Cells"),
            Patch(facecolor=BIN_COLOR,  edgecolor="black", label="Bins"),
        ]
        plt.legend(handles=legend, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, title="Layers")
        plt.tight_layout()
        plt.savefig(os.path.join(sample_folder, "alignment_visualization.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        adata_cellbin.write_h5ad(cellbin_out)
        adata_xenium.write_h5ad(cell_out)
        print(f'  finished {sample}')

    # ── Part B: Batch file creation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part B: Xenium Batch Creation")
    print("=" * 60)

    os.makedirs(XEN_OUT_DIR, exist_ok=True)

    # Load reference parameters
    print("\nLoading reference parameters...")
    ref_params = xen_load_visium_reference(XEN_ANNO_NAMES_PATH, XEN_VISIUM_BATCH_DIR)
    print(f"  Classes ({len(ref_params['class_names'])}): {ref_params['class_names']}")
    if 'max_tokens' in ref_params:
        print(f"  Max tokens (T): {ref_params['max_tokens']}")
        print(f"  Feature dim (C): {ref_params['feature_dim']}")

    # Determine domain offset
    print("\nDetermining Xenium domain IDs...")
    max_visium_domain = xen_get_max_visium_domain_id(XEN_VISIUM_BATCH_DIR)
    xenium_domain_start = max_visium_domain + 1
    print(f"  Xenium domain IDs will start from: {xenium_domain_start}")
    print(f"\nRadius mode: FIXED = {XEN_FIXED_RADIUS} px")

    # Process each sample
    batch_mapping = {}
    for idx, sample in enumerate(samples):
        domain_id = xenium_domain_start + idx
        print(f"\n{'=' * 60}")
        print(f"Processing: {sample} (batch_xen_{idx:03d}, domain={domain_id})")
        print(f"{'=' * 60}")

        processed_path = os.path.join(XEN_PROCESSED_ROOT, sample)

        print("Loading Xenium data...")
        adata_cell, adata_cellbin = xen_load_xenium_data(processed_path)
        print(f"  Cells after QC: {adata_cell.n_obs}")

        print("Creating label matrix...")
        Y_all, valid_mask, labels = xen_create_label_matrix(
            adata_cell, ref_params['class_to_idx'], cell_type_mapping
        )
        print(f"  Valid labeled cells: {valid_mask.sum()} / {adata_cell.n_obs}")

        print("Mapping bins to cells...")
        df_pairs, mapping_stats = xen_compute_bin_to_cell_mapping(
            adata_cell, adata_cellbin, fixed_radius=XEN_FIXED_RADIUS
        )
        print(f"  Bin spacing: {mapping_stats['px_per_bin']:.2f} px")
        print(f"  Cells with bins: {mapping_stats['cells_with_bins']} / {mapping_stats['total_cells']}")
        print(f"  Bins per cell: median={mapping_stats['bins_per_cell_median']:.0f}, "
              f"mean={mapping_stats['bins_per_cell_mean']:.1f}")

        target_size = ref_params.get('batch_size', valid_mask.sum())
        print(f"Selecting cells (target: {target_size})...")
        selected_cells = xen_stratified_sample_cells(
            df_pairs, valid_mask, labels, target_size, XEN_SEED
        )

        max_tokens = ref_params.get('max_tokens', 100)
        print("Extracting features...")
        X = xen_extract_features_for_cells(
            selected_cells, df_pairs, adata_cellbin, max_tokens, XEN_SEED
        )
        Y = Y_all[selected_cells]

        batch_name = f"batch_xen_{idx:03d}"
        print(f"Saving {batch_name} with domain_id={domain_id}...")
        xen_save_batch(X, Y, domain_id, batch_name, XEN_OUT_DIR)

        if hasattr(adata_cellbin, 'file'):
            adata_cellbin.file.close()
        del adata_cell, adata_cellbin, df_pairs, X, Y_all
        gc.collect()

        batch_mapping[batch_name] = {
            'sample_name': sample,
            'n_cells': len(selected_cells),
            'domain_id': domain_id,
            'px_per_bin': mapping_stats['px_per_bin'],
            'radius': mapping_stats['radius'],
        }

    # Save batch-sample mapping
    mapping_path = os.path.join(XEN_OUT_DIR, "batch_sample_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(batch_mapping, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for batch_name, info in batch_mapping.items():
        print(f"  {batch_name}: {info['sample_name']} "
              f"(n={info['n_cells']}, domain={info['domain_id']})")
    print(f"\nSaved mapping: {mapping_path}")
    print("Processing complete!")


# =========================================================================
# MODE DISPATCH
# =========================================================================

if _known.mode == 'visium':
    _run_visium()
elif _known.mode == 'xenium':
    _run_xenium()
