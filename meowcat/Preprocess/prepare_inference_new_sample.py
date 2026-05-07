# prepare_inference_inputs_new_sample.py

import os
import numpy as np
import numpy.lib.format as npy_fmt
import pandas as pd
from PIL import Image
import scanpy as sc
import pickle
import json
import sys
import warnings
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ExtractFeatures'))
from UTILS import *

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

def main(base_path, sample):
    print(f"Preparing inference inputs for sample: {sample}")
    sample_path = os.path.join(base_path, sample)
    os.makedirs(sample_path, exist_ok=True)

    # ------------------ 1. Image feature ------------------
    prefix = os.path.join(sample_path, "he")   # or "he" depending on naming
    image_path = get_image_filename(prefix)

    # Get image dimensions without loading full image into RAM.
    # Use PIL for formats it supports; fall back to load_image for others.
    try:
        with Image.open(image_path) as pil_img:
            width, height = pil_img.size          # (W, H)
        print(f"height: {height} width: {width}  (PIL header-only)")
    except Exception:
        img = load_image(image_path, verbose=True)
        height, width = img.shape[:2]
        del img
        print(f"height: {height} width: {width}  (full load fallback)")


    adata_img = sc.read(os.path.join(sample_path, "single_super_emb.h5ad"))
    patch_size = 16
    feature_dim = adata_img.shape[1]
    grid_w = width // patch_size
    grid_h = height // patch_size
    print(f"grid: {grid_h} x {grid_w}, feature_dim: {feature_dim}")

    shape = (grid_h, grid_w, feature_dim)
    nbytes = grid_h * grid_w * feature_dim * 4
    GB = nbytes / (1024 ** 3)

    # For large grids (>4 GB), use memory-mapped .npy to avoid OOM.
    # For small grids, use the original in-memory pickle for compatibility.
    if GB > 4.0:
        print(f"Large grid ({GB:.1f} GB) — writing memory-mapped .npy")
        npy_path = os.path.join(sample_path, "embeddings-hist.npy")
        with open(npy_path, 'wb') as f:
            npy_fmt.write_array_header_2_0(f,
                {'descr': np.dtype('float32').str,
                 'fortran_order': False, 'shape': shape})
            header_offset = f.tell()

        feature_array = np.memmap(npy_path, dtype='float32', mode='r+',
                                  shape=shape, offset=header_offset)
        feature_array[:] = 0

        coords = adata_img.obsm['spatial']
        for i, (x, y) in enumerate(coords):
            if 0 <= y < grid_h and 0 <= x < grid_w:
                feature_array[y, x, :] = adata_img.X[i]

        feature_array.flush()
        del feature_array
        print(f"Saved {npy_path}")
    else:
        print(f"Small grid ({GB:.1f} GB) — writing pickle")
        feature_array = np.zeros(shape, dtype=np.float32)

        coords = adata_img.obsm['spatial']
        for i, (x, y) in enumerate(coords):
            if 0 <= y < grid_h and 0 <= x < grid_w:
                feature_array[y, x, :] = adata_img.X[i]

        with open(os.path.join(sample_path, "embeddings-hist.pickle"), "wb") as f:
            pickle.dump(feature_array, f)
        print(f"Saved embeddings-hist.pickle")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_inference_inputs.py <base_path> <sample>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
