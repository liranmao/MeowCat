# prepare_inference_inputs.py

import os
import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
import pickle
import json
import sys
import warnings

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None 

def main(base_path, sample):
    print(f"Preparing inference inputs for sample: {sample}")
    sample_path = os.path.join(base_path, sample)
    os.makedirs(sample_path, exist_ok=True)

    # ------------------ 1. Image feature ------------------
    image_path = os.path.join(sample_path, "he.jpg")
    with Image.open(image_path) as img:
        width, height = img.size


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

    # ------------------ 2. Cell type matrix ------------------
    df_pred = pd.read_csv(os.path.join(sample_path, "tangram", "tangram_ct_pred.csv"), index_col=0)

    with open(os.path.join(sample_path, "anno-names.txt"), "w") as f:
        for ct in df_pred.columns:
            f.write(f"{ct}\n")

    df_normalized_reset = df_pred.reset_index()
    df_normalized_reset.rename(columns={"index": "spot"}, inplace=True)
    df_normalized_reset.to_csv(os.path.join(sample_path, "anno_matrix.tsv"), sep="\t", index=False)

    # ------------------ 3. Spatial location ------------------
    adata = sc.read_visium(sample_path)
    coords = adata.obsm['spatial']
    barcodes = adata.obs_names
    df_locs = pd.DataFrame({
        'spot': barcodes,
        'x': coords[:, 0].astype(int),
        'y': coords[:, 1].astype(int)
    })
    df_locs.to_csv(os.path.join(sample_path, "locs.tsv"), sep='\t', index=False)

    # ------------------ 4. JSON metadata ------------------
    meta_path = os.path.join(sample_path, "spatial", "scalefactors_json.json")
    with open(meta_path, "r") as f:
        data = json.load(f)

    with open(os.path.join(sample_path, "radius-raw.txt"), "w") as f:
        dia = data['spot_diameter_fullres'] / 2
        f.write(f"{dia}")

    with open(os.path.join(sample_path, "pixel-size-raw.txt"), "w") as f:
        f.write(f"0.2513")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_inference_inputs.py <base_path> <sample>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])