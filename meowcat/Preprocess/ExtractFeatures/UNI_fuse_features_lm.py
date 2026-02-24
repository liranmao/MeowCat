import os 
import argparse
from UTILS import *
import torch 
from multiprocessing import cpu_count
import pdb
from time import time 
import scanpy as sc 

# define parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_global_path', type=str, required=True,help="where the global embedding is saved ")
    parser.add_argument('--read_local_path', type=str, required=True,help="where the local embedding is saved ")
    parser.add_argument('--sample',type=str,required=True, help ="unique ID to idenify dataset")
    parser.add_argument('--save_dir', type=str, required=True, help="where the embedding will be saved")
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='single')
    parser.add_argument('--device', type=str, default='cuda:0',help="which GPU will be used")
    parser.add_argument('--batch_size', type=int, default=128,help="the batch size for UNI inference at a time")
    parser.add_argument('--patch_size', type=int, default=224,help="size of image tiles for UNI input")
    parser.add_argument('--token_size', type=int, default=16,help="size of super pixel")
    parser.add_argument('--stride', type=int, default=112,help="length of sliding window")
    parser.add_argument('--clustering', action='store_true', help='whether to cluster for embedding')
    parser.add_argument('--use_cuML', action='store_true', help='Use cuML for acceleration')
    parser.add_argument('--input_format', type=str, choices=['h5ad', 'rds',"npy"], default='h5ad')
    return parser.parse_args()


#extract fused feature
def extract_fused_features(
    image_dict:dict =None,
    global_emb:np.ndarray = None,
    local_emb:np.ndarray =None,
    coords:np.ndarray =None,
    grid_coords:np.ndarray=None,
    patch_size:int = 224,
    stride: int = 112,
    token_size: int = 16,
) -> np.ndarray:

    H_tokens,W_tokens,H_out,W_out = image_dict["H_tokens"],image_dict["W_tokens"],image_dict["H_out"],image_dict["W_out"]
    global_dim = global_emb.shape[-1]
    local_dim = local_emb.shape[-1]
    D_fused = global_dim + local_dim 
    feature_map = torch.zeros((H_tokens, W_tokens, D_fused), dtype=torch.float16, device="cpu")
    count_map = torch.zeros((H_tokens, W_tokens), dtype=torch.float16, device="cpu")
    # Precompute center weights (shared by all patches)
    center_weight = torch.tensor(get_center_weights(size=14), dtype=torch.float32, device="cpu")  # [14,14]
    #pdb.set_trace()
    for i in range(len(coords)):
#         batch = batch.to(device, non_blocking=True)
#         cls_tokens, local_tokens = extract_features(model, batch)  # [B, D], [B, 196, D]
        cls_tokens = torch.from_numpy(global_emb[i])
        local_tokens = torch.from_numpy(local_emb[i])
        cls_tokens = cls_tokens.unsqueeze(0)
        local_tokens =local_tokens.unsqueeze(0)
        fused = torch.cat([
            cls_tokens[:, None, None, :].expand(-1, 14, 14, -1),
            local_tokens.view(-1, 14, 14, local_dim)
        ], dim=-1)  # shape: [B, 14, 14, D_fused]
        tops,lefts = coords[i]
        top_idx = tops // token_size
        left_idx = lefts // token_size
        #pdb.set_trace()
        # Center-weighted fusion
        fused = fused.squeeze(0)
        weighted = fused.cpu() * center_weight[..., None]  # [14, 14, D_fused]
        feature_map[top_idx:top_idx + 14, left_idx:left_idx + 14] += weighted
        count_map[top_idx:top_idx + 14, left_idx:left_idx + 14] += center_weight

    # Normalize final feature map
    feature_map /= count_map.unsqueeze(-1).clamp(min=1e-6)

    # Crop to original image region (remove padding)
    start_y = (patch_size // 2) // token_size
    start_x = (patch_size // 2) // token_size

    final_map = feature_map[start_y:start_y + H_out, start_x:start_x + W_out]
    return final_map.detach().cpu().numpy()  # [H_out, W_out, D_fused]
    

@torch.inference_mode()
@monitor_cuda_memory()
def main(args):
    ############## print remote server configuration ################
    ############## print remote server configuration ################
    ############## print remote server configuration ################
    print("✅ CUDA is available：", torch.cuda.is_available())
    print("✅ Count of GPUs：",torch.cuda.device_count())
    print("✅ Current device：", torch.cuda.current_device())
    print("✅ Name of device：", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("✅ Count of cpus:",cpu_count())
    
    ############# for A100: run fast 
    ############# for A100: run fast 
    ############# for A100: run fast 
    torch.set_float32_matmul_precision('high')

    ############# read image file, mask file(generated from HistoSweep) and create save dir #######################
    t0 = time()
    args.mask_path = args.read_local_path + f"/mask/mask-small.png"
    print(args.mask_path)
    if(not os.path.exists(f"{args.mask_path}")):
        print("mask file don't exist")
        exit(1)
    full_grid_mask = np.array(Image.open(args.mask_path)) > 0  # Convert to binary mask
    print(f"grid mask.shape ={full_grid_mask.shape}")
    
    
    if(args.mode == "single"):
        if(args.input_format == "npy"):
            global_emb = np.load(args.read_global_path + "global_emb.npy")
        elif(args.input_format == "h5ad"):
            adata = sc.read(args.read_global_path + "global_emb.h5ad")
            global_emb = adata.X
        elif(args.input_format =="rds"):
            import pyreadr
            global_emb = pyreadr.read_r(args.read_global_path + "global_emb.rds")
        else:
            print("the input format is wrong, please select one of [npy,h5ad,rds]")
            exit(1)
    elif(args.mode == "multi"):
        if(args.input_format == "npy"):
            global_emb = np.load(args.read_global_path + "global_harmony.npy")
        elif(args.input_format == "h5ad"):
            adata = sc.read(args.read_global_path + "global_harmony.h5ad")
            global_emb = adata.obsm["X_harmony"]
        elif(args.input_format =="rds"):
            import pyreadr
            global_emb = pyreadr.read_r(args.read_global_path + "global_harmony.rds")
        else:
            print("the input format is wrong, please select one of [npy,h5ad,rds]")
            exit(1)
    else:
        print("mode should one of choice in [single,multi]")
        exit(1)
        
    local_emb = np.load(args.read_local_path + "local_emb.npy") 
    coords = np.load(args.read_local_path + "coords.npy") 
    grid_coords = np.load(args.read_local_path + "grid_coords.npy") 
    
    with open(args.read_local_path + "image.txt", "r") as f:
        lines = f.readlines()
    values = lines[1].strip().split()  
    W_pad,H_pad,W_tokens,H_tokens,W_ori,H_ori,W_out,H_out = map(int, values)
    image_dict ={"W_pad":W_pad,"H_pad":H_pad,"W_tokens":W_tokens,"H_tokens":H_tokens,
                "W_ori":W_ori,"H_ori":H_ori,"W_out":W_out,"H_out":H_out}
    
    print(f"mode ={args.mode} ,global_emb.shape = {global_emb.shape}")
    print(f"mode ={args.mode} ,local_emb.shape = {local_emb.shape}")
    print(f"mode ={args.mode} ,coords.shape = {coords.shape}")
    print(f"mode ={args.mode} ,grid_coords.shape = {grid_coords.shape}")
    print(f"mode ={args.mode}, W_pad ={W_pad},H_pad = {H_pad}")
    print(f"mode ={args.mode}, W_tokens ={W_tokens},H_tokens = {H_tokens}")
    print(f"mode ={args.mode}, W_ori ={W_ori},H_ori = {H_ori}")
    print(f"mode ={args.mode}, W_out ={W_out},H_out = {H_out}")
    
    t0 = time()
    final_map = extract_fused_features(
        image_dict = image_dict,
        global_emb = global_emb,
        local_emb = local_emb,
        coords = coords,
        grid_coords = grid_coords,
        patch_size=args.patch_size,
        stride=args.stride,
        token_size= args.token_size
    )
    print(f"final_map.dtype = {final_map.dtype}")
    print(f"final_map.shape={final_map.shape}")  
    t1 =time()
    print(f'extract fused features cost {int(t1-t0)}s!!!')
    
    del local_emb
    del global_emb
    del coords
    del grid_coords
    import gc
    gc.collect()
     
#     t0 = time()
#     token_grid_coords = []
#     token_features = []
#     for i in range(full_grid_mask.shape[0]):
#         for j in range(full_grid_mask.shape[1]):
#             if full_grid_mask[i, j]:
#                 token_grid_coords.append((i, j))
#                 token_features.append(final_map[i, j])
#     t1 = time()
    t0 = time()
    mask_indices = np.argwhere(full_grid_mask)
    token_grid_coords = mask_indices
    token_features = final_map[full_grid_mask].astype(np.float16)
    t1 = time()
    print(f'filter 16*16 super pixel embedding costs {int(t1 - t0)}s!!!')
    print(f'filter 16*16 super pixel embedding costs {int(t1-t0)}s!!!')
    
    del final_map
    import gc 
    gc.collect()
    
    ################ save 16*16 super pixel embeddings to h5ad file ######################
    t0 = time()
    adata = sc.AnnData(X=np.array(token_features,dtype=np.float16))
    token_grid_coords = np.array(token_grid_coords)
    adata.obsm["spatial"] = token_grid_coords[:, [1, 0]]
    adata.obs["sample"] = args.sample # need to set for Visulization
       
    print(adata)
    
    if args.clustering:
        print("\n")
        print("✅ 3. Clustering super pixel embedding....\n")

        if args.use_cuML:
            if(adata.shape[0]<3e6):
                print("Use cuML library to calculate PCA and Kmeans...")
                import cupy as cp
                import cudf
                from cuml.decomposition import PCA
                from cuml.manifold import UMAP as cuUMAP
                from cuml.cluster import KMeans as cuKMeans
                import random
                import cuml
                random.seed(42)
                np.random.seed(42)
                start_pca = time()
                X_cp = cp.asarray(adata.X)
                pca = PCA(n_components=50,random_state=42)
                X_cp = X_cp.astype('float32')
                X_pca = pca.fit_transform(X_cp)
                adata.obsm["X_pca"] = X_pca.get()
                end_pca = time()
                print(f"perform PCA dimension reduction costs {int(end_pca - start_pca)} s")
                start_kmeans = time()
                ncluster_list = [5, 10, 15, 20, 25, 30]
                for cluster in ncluster_list:
                    print(f"clustering with Kmeans(K={cluster})")
                    t0 = time()
                    kmeans_cuml = cuKMeans(init="k-means||", n_clusters=cluster, random_state=42)
                    kmeans_cuml.fit(X_pca)
                    cluster_labels = kmeans_cuml.labels_.get()
                    adata.obs["kmeans_{}".format(cluster)] = cluster_labels.astype(str)
                end_kmeans = time()
                print(f"perform Kmeans costs {int(end_kmeans - start_kmeans)} s") 
                
            else:
                import cupy as cp
                import cudf
                #from cuml.decomposition import PCA
                from cuml.decomposition import IncrementalPCA
                from cuml.manifold import UMAP as cuUMAP
                from cuml.cluster import KMeans as cuKMeans
                import random
                import cuml
                # PCA
                # PCA settings
                n_components = 200
                batch_size = 50000

                # Initialize Incremental PCA
                ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

                t0 = time()

                # Step 1: Fit the Incremental PCA in batches
                for i in range(0, adata.n_obs, batch_size):
                    # Move one batch from CPU to GPU
                    X_batch = cp.asarray(adata.X[i:i + batch_size].astype(np.float32))

                    # Perform partial_fit on the GPU
                    ipca.partial_fit(X_batch)

                    # Free GPU memory
                    del X_batch
                    cp._default_memory_pool.free_all_blocks()

                # Step 2: Transform the full dataset in batches
                X_pca_chunks = []
                for i in range(0, adata.n_obs, batch_size):
                    # Load next batch
                    X_batch = cp.asarray(adata.X[i:i + batch_size].astype(np.float32))

                    # Perform PCA transformation
                    X_pca_batch = ipca.transform(X_batch)

                    # Move result back to CPU and store
                    X_pca_chunks.append(X_pca_batch.get())

                    # Free GPU memory
                    del X_batch, X_pca_batch
                    cp._default_memory_pool.free_all_blocks()

                # Concatenate all transformed batches on CPU
                X_pca = np.vstack(X_pca_chunks)

                t1 = time()
                print(f"Incremental PCA completed in {int(t1 - t0)} seconds. Shape = {X_pca.shape}")

                adata.obsm["X_pca"] = X_pca
                X_pca = cp.asarray(X_pca)
                
                start_kmeans = time()
                ncluster_list = [5, 10, 15, 20, 25, 30]
                for cluster in ncluster_list:
                    print(f"clustering with Kmeans(K={cluster})")
                    t0 = time()
                    kmeans_cuml = cuKMeans(init="k-means||", n_clusters=cluster, random_state=42)
                    kmeans_cuml.fit(X_pca)
                    cluster_labels = kmeans_cuml.labels_.get()
                    adata.obs["kmeans_{}".format(cluster)] = cluster_labels.astype(str)
                end_kmeans = time()
                print(f"perform Kmeans costs {int(end_kmeans - start_kmeans)} s") 
        else:
            print("Use Sklearn library to calculate PCA and Kmeans...")
            # perform PCA dimension reduction 
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.cluster import MiniBatchKMeans

            start_pca = time()
            ncluster_list = [5,10,15,20,25,30]
            pca = PCA(n_components=50)
            PCA_emb = pca.fit_transform(adata.X)
            total_variance = pca.explained_variance_ratio_.sum()
            print(f"total_variance_ratio: {total_variance:.4f}")
            end_pca = time()
            print(f"perform PCA dimension reduction costs {int(end_pca - start_pca)} s")
            adata.obsm["X_pca"] = PCA_emb

            # perform kmeans clusering
            if(adata.X.shape[0]< 1e7):
                start_kmeans = time()
                for cluster in ncluster_list:
                    print(f"clustering with Kmeans(K={cluster})")
                    kmeans = KMeans(n_clusters=cluster, random_state=42)
                    cluster_labels = kmeans.fit_predict(PCA_emb)
                    adata.obs["kmeans_{}".format(cluster)] = cluster_labels.astype(str)
                end_kmeans = time()
                print(f"perform Kmeans costs {int(end_kmeans - start_kmeans)} s")
            else:
                start_kmeans = time()
                batch_size = 1000
                ncluster_list = [5, 10, 15, 20, 25, 30]
                for cluster in ncluster_list:
                    print(f"clustering with MiniBatchKMeans(K={cluster})")
                    kmeans = MiniBatchKMeans(n_clusters=cluster, random_state=42, batch_size=batch_size)
                    cluster_labels = kmeans.fit_predict(PCA_emb)
                    adata.obs[f"kmeans_{cluster}"] = cluster_labels.astype(str)    
                end_kmeans = time()
                print(f"perform miniBatchKmeans costs {int(end_kmeans - start_kmeans)} s")
        print(adata)
        
    print("\n")
    print("✅ 4. save embedding and coordinate to h5ad...")
    # save h5ad

    if(args.mode == "single"):
        t0 = time()
        h5ad_path = os.path.join(args.save_dir, f"single_super_emb.h5ad")
        if(adata.shape[0]>3e6):
            adata.write(h5ad_path, compression="lzf")
        else:
            adata.write(h5ad_path, compression="gzip")
        t1 = time()
        print(f"Saved embedding to {h5ad_path} with shape {adata.shape} cost {int(t1-t0)}s")
        print("\n")
    if(args.mode == "multi"):
        t0 = time()
        h5ad_path = os.path.join(args.save_dir, f"multi_super_emb.h5ad")
        if(adata.shape[0]>3e6):
            adata.write(h5ad_path, compression="lzf")
        else:
            adata.write(h5ad_path, compression="gzip")
        t1 = time()
        print(f"Saved embedding to {h5ad_path} with shape {adata.shape} cost {int(t1-t0)}s")
        print("\n")
        
    
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉 Congratulations, Pipeline completed sucessfully!!!!🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")



if __name__ == '__main__':
    args = get_args()
    print(args)    
    
    monitor = CPUMemoryMonitor(interval=0.1)
    monitor.start()
    start_time = time()
    log_system_info("START")
    t0 = time()
    main(args)
    t1 = time()
    print(f"All steps cost {t1 - t0}s!!!!")
    log_system_info("END")
    print(f"[INFO-END] Total runtime: {time() - start_time:.2f} seconds")
    monitor.stop()