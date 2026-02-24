import scanpy as sc 
from harmony import harmonize
import argparse
from time import time
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from time import time
import os 
from UTILS import load_and_concat_data

#python ExtractFeatures/RunHarmonyPy1.py --read_dir ./result/ --save_dir ./result/ --sample_list H1_low G2_low >./Log/log_H1_low_G2_low.txt 2>&1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, default='BBBB',
                       help='dictionary to read dataset')
    parser.add_argument('--save_dir', type=str, default='BBBB',
                       help='Directory to save visualization results')
    parser.add_argument('--sample_list',nargs = '+')
    parser.add_argument('--n_comps',type=int,default =100)
    parser.add_argument('--use_cuML', action='store_true', help='Use cuML for acceleration')
    return parser.parse_args()
        
def harmony_cuML(adata, args):
    """
    Process and visualize the concatenated data.
    
    Args:
        adata (AnnData): Concatenated AnnData object
        save_dir (str): Directory to save visualization results
        fig_size (tuple): Figure size for the plots
    """
    print("Processing data...")
    # Preprocess data
    import cupy as cp
    import cudf
    from cuml.decomposition import PCA
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import KMeans as cuKMeans
    import random
    import cuml
    
    # PCA
    t0 = time()
    X_cp = cp.asarray(adata.X.copy())
    pca = PCA(n_components=args.n_comps,random_state=42)
    X_pca = pca.fit_transform(X_cp)
    adata.obsm["X_pca"] = X_pca.get()
    t1 = time()
    print(f"Calculate PCA cost {t1-t0}s!!!")
    
    batch_key = "sample"
    print(f"Running Harmony with cuML using batch key: {batch_key}")
    
    Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key =batch_key,,use_gpu =True)
    #Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key =batch_key)
    adata.obsm['X_harmony'] = Z
    # X_cp = cp.asarray(adata.X.copy())
    X_harmony = cp.asarray(adata.obsm['X_harmony'])
    # UMAP
    t0 = time()
    cu_umap_model = cuUMAP(n_components=2,init="spectral",random_state=42)
    umap_emb = cu_umap_model.fit_transform(X_harmony)
    adata.obsm["X_umap"] = umap_emb.get()
    t1 = time()
    print(f"Calculate UMAP cost {t1-t0}s!!!")
    ## should perserve umap plot
    fig, axs = plt.subplots(1, 1)
    sc.pl.umap(adata, color="sample", ax=axs, title="UMAP by Sample", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"umap_plot.png"), bbox_inches="tight")
    plt.close()
    
    print(adata.obsm['X_harmony'].shape)
    for sample in args.sample_list:
        ad = adata[adata.obs["sample"] == sample].copy()
        print(ad)
        ad.write(f"{args.save_dir}/{sample}/global_harmony.h5ad",compression="gzip")
    
def harmony_cpu(adata,args):
    print("run harmony with cpu...")
    print(adata)
    sc.tl.pca(adata,n_comps = args.n_comps)

    #Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key =batch_key)
    Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key =batch_key,,use_gpu =True)
    adata.obsm['X_harmony'] = Z
    
    print(adata.obsm['X_harmony'].shape)
    for sample in args.sample_list:
        ad = adata[adata.obs["sample"] == sample].copy()
        print(ad)
        ad.write(f"{args.save_dir}/{sample}/global_harmony.h5ad",compression="gzip")
    
def main(args):
    t0 = time()
    # Read sample paths from file
    if not os.path.exists(args.read_dir):
        raise ValueError(f"Path file {args.path_file} does not exist!")
    
    sample_paths = []
    for sample in args.sample_list:
        file_path = args.read_dir +"/" + sample + "/global_emb.h5ad"
        sample_paths.append(file_path)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    #delete_all_files(args.save_dir)
    
    print(f"Processing {len(sample_paths)} samples...")
    
    # Load and concatenate data
    t0 = time()
    adata_concat = load_and_concat_data(sample_paths)
    print(adata_concat)
    t1 = time()
    print(f"read and concat adata cost {t1 - t0}s!!!")
    
    if args.use_cuML:
        harmony_cuML(adata_concat, args)
    else:
        harmony_cpu(adata_concat, args)
    
    
if __name__ == '__main__':
    args = get_args()
    print(args)    
    start_time = time()
    main(args)
    print(f"[INFO-END] Total runtime: {time() - start_time:.2f} seconds")