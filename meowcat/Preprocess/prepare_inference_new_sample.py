# prepare_inference_inputs_new_sample.py

import os
import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
import pickle
import json
import sys
import warnings
sys.path.append('./ExtractFeatures/') 
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

    img = load_image(image_path, verbose=True)

    # get width / height
    height, width = img.shape[:2]
    print("height:", height, "width:", width)


    adata_img = sc.read(os.path.join(sample_path, "single_super_emb.h5ad"))
    patch_size = 16
    feature_dim = adata_img.shape[1]
    grid_w = width // patch_size
    grid_h = height // patch_size
    feature_array = np.zeros((grid_h, grid_w, feature_dim), dtype=np.float32)

    coords = adata_img.obsm['spatial']
    for i, (x, y) in enumerate(coords):
        if 0 <= y < grid_h and 0 <= x < grid_w:
            feature_array[y, x, :] = adata_img.X[i]

    with open(os.path.join(sample_path, "embeddings-hist.pickle"), "wb") as f:
        pickle.dump(feature_array, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_inference_inputs.py <base_path> <sample>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
