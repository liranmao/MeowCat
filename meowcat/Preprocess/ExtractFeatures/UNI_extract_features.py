import os
from UTILS import *
import torch
from torchvision import transforms
import timm
import numpy as np
import pandas as pd 
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
import scanpy as sc 
from PIL import Image, ImageOps
import pdb
import psutil
import platform
import threading
from datetime import datetime
from multiprocessing import cpu_count
from time import time
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
import pdb
import signal
# Register signal handlers for SIGINT (Ctrl+C) and SIGTERM (kill)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


# define parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=True,help="where the image is saved ")
    parser.add_argument('--sample',type=str,required=True, help ="unique ID to idenify dataset")
    parser.add_argument('--save_dir', type=str, required=True, help="where the embedding will be saved")
    parser.add_argument('--weight_dir', type=str, default="/project/MultiSampleIstar/yxk/Foundationmodel/UNI/assets/ckpts/"
    "vit_large_patch16_224.dinov2.uni_mass100k/"
    "pytorch_model.bin", help="Path to load the UNI model weights")
    parser.add_argument('--device', type=str, default='cuda:0',help="which GPU will be used")
    parser.add_argument('--batch_size', type=int, default=128,help="the batch size for UNI inference at a time")
    parser.add_argument('--patch_size', type=int, default=224,help="size of image tiles for UNI input")
    parser.add_argument('--token_size', type=int, default=16,help="size of super pixel")
    parser.add_argument('--stride', type=int, default=112,help="length of sliding window")
    parser.add_argument('--num_workers', type=int, default=8,help="Number of worker processes used for data loading")
    parser.add_argument('--clustering', action='store_true', help='whether to cluster for embedding')
    parser.add_argument('--use_cuML', action='store_true', help='Use cuML for acceleration')
    parser.add_argument('--output_format', type=str, choices=['h5ad', 'rds',"npy"], default='h5ad')
    return parser.parse_args()

# create model
def create_model(ckpt_path: str) -> torch.nn.Module:
    """Create and load the ViT model."""
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        global_pool='',
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    return model


#Feature Extraction
@torch.inference_mode()
def extract_features(model: torch.nn.Module, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #final_output, _ = model.forward_intermediates(batch, return_prefix_tokens=False)
    #final_output = model(batch)
    with torch.cuda.amp.autocast():
        final_output = model(batch)
    local_emb = final_output[:, 1:]  # shape [B, 196, D]
    global_emb = final_output[:, 0]  # shape [B, D]
    return global_emb, local_emb

    
# SlidingWindowDataset with mask
class SlidingWindowDataset(Dataset):
    def __init__(self, image: Image.Image, mask: np.ndarray, patch_size: int = 224, stride: int = 112):
        self.image = ImageOps.expand(image, border=patch_size // 2, fill=0)
        self.mask = np.pad(
            mask,
            pad_width=((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2)),
            mode='constant',
            constant_values=0
        )
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        self.coords = []
        self.grid_coords = []
        W, H = self.image.size

        print(f"Generating from full image(W={W},H={H}) sliding window coordinates with patch_size={patch_size}, stride={stride}...")

        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                mask_patch = self.mask[top:top + patch_size, left:left + patch_size]
                if np.any(mask_patch):
                    #print(f"top ={top},left={left}")
                    #pdb.set_trace()
                    self.coords.append((top, left))
                    self.grid_coords.append((top/stride,left/stride))

        print(f"✅ Total selected patches covering mask region: {len(self.coords)}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        #pdb.set_trace()
        top, left = self.coords[idx]
        patch = self.image.crop((left, top, left + self.patch_size, top + self.patch_size))
        patch_tensor = self.transform(patch)
        return patch_tensor, top, left, idx
    
    
#patch_embeddings.append(feature_emb.detach().cpu().numpy())    
# extract global_embeding and local embedding    
@torch.inference_mode()
def extract_global_local_feature(
    model: torch.nn.Module,
    image: Image.Image,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    patch_size: int = 224,
    stride: int = 112,
    token_size: int = 16,
    num_workers: int = 4,
    save_dir = None,
    sample = None,
    output_format = None
) -> np.ndarray:
    model.eval()
    dataset = SlidingWindowDataset(image,mask,patch_size=patch_size, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print("Total number of 224*224*3 image tiles:", len(dataset))
    
    H_ori, W_ori = image.size[1], image.size[0]
    H_out = H_ori // token_size
    W_out = W_ori // token_size
    print(f"Original image shape:({H_ori},{W_ori})")
    print(f"Original token shape:({H_out},{W_out})")
    
    #pdb.set_trace()
    W_pad, H_pad = dataset.image.size
    print(f"Padded image shape:({H_pad},{W_pad})")
    H_tokens = H_pad // token_size
    W_tokens = W_pad // token_size
    print(f"Padded token size:({H_tokens},{W_tokens})")
    
    # assume D = 1024
    D = 1024
    D_fused = D  # final dimension：don't concatinate

    # Initialize feature maps
    #feature_map = torch.zeros((H_tokens, W_tokens, D_fused), dtype=torch.float16, device="cpu")
    #count_map = torch.zeros((H_tokens, W_tokens), dtype=torch.float16, device="cpu")

    # Center weight matrix: center weight for each patch in a sliding window
    #center_weight = torch.tensor(get_center_weights(size=14), dtype=torch.float32, device="cpu")  # shape [14,14]
    #spatial_weight = center_weight  #     

    local_emb =  np.empty((len(dataset), 14, 14, D), dtype=np.float16)
    global_emb = np.empty((len(dataset), D), dtype=np.float16)
    ptr = 0 
    for batch, tops, lefts, idx in tqdm(loader, desc="Sliding Window Inference"):
        batch = batch.to(device, non_blocking=True)
        B = batch.shape[0]
        cls_tokens, local_tokens = extract_features(model, batch)  # cls_tokens: [B,D], local_tokens: [B,196,D]
        #pdb.set_trace()
        # Expand shapes 
        #global_expanded = cls_tokens[:, None, None, :].expand(-1, 14, 14, -1)  # [B,14,14,D]
        local_reshaped = local_tokens.view(-1, 14, 14, D)                      # [B,14,14,D]
        global_emb[ptr:ptr + B] = cls_tokens.detach().cpu().numpy().astype(np.float16)
        local_emb[ptr:ptr + B] = local_reshaped.detach().cpu().numpy().astype(np.float16)
        ptr += B
        #idx_list.extend(idx.detach().cpu().numpy())
    #pdb.set_trace()
    #local_emb = np.concatenate(local_emb, axis=0)
    print(f"local_emb.shape = {local_emb.shape}")
    np.save(save_dir+"local_emb.npy", local_emb)

    coords = np.array(dataset.coords)
    print(f"coord.shape = {coords.shape}")
    np.save(save_dir+"coords.npy", coords)

    grid_coords = np.array(dataset.grid_coords)
    print(f"grid_coords.shape = {grid_coords.shape}")
    np.save(save_dir+"grid_coords.npy", grid_coords)
    
    # I need to record W_pad,H_pad，W_tokens,H_tokens
    with open(save_dir + "image.txt", "w") as f:
        f.write("W_pad H_pad W_tokens H_tokens W_ori H_ori W_out H_out \n")
        f.write(f"{W_pad} {H_pad} {W_tokens} {H_tokens} {W_ori} {H_ori} {W_out} {H_out}\n")
    
    if(output_format == "npy"):
        #global_emb = np.concatenate(global_emb, axis=0)
        print(f"global_emb.shape ={global_emb.shape}")
        np.save(save_dir+"global_emb.npy", global_emb)
        
    #global_emb = np.concatenate(global_emb, axis=0)
    print(f"global_emb.shape = {global_emb.shape}")
    if(output_format == "h5ad"):
        ## save global_emb
        adata = sc.AnnData(global_emb)
        grid_coords = dataset.grid_coords 
        adata.obsm["spatial"] = np.array(grid_coords)[:, [1, 0]]  # swap to [x, y]
        adata.obs["sample"] = args.sample
        print(adata)
        t0 = time()
        h5ad_path = os.path.join(save_dir, f"global_emb.h5ad")
        adata.write(h5ad_path, compression="gzip")
        t1 = time()
        print(f"Saved embedding to {h5ad_path} with shape {adata.shape} cost {int(t1-t0)}s")
        ## save local_emb
        ####################################################################################
        ####################################################################################
        
    elif(output_format =="rds"):
        import pyreadr
        #Ionly need to save PCA 
        import cupy as cp
        import cudf
        from cuml.decomposition import PCA
        from cuml.manifold import UMAP as cuUMAP
        from cuml.cluster import KMeans as cuKMeans
        import random
        import cuml
        X_cp = cp.asarray(global_emb.astype(np.float32))
        pca = PCA(n_components=200, random_state=42)
        X_pca = pca.fit_transform(X_cp)
        global_emb_pca = pd.DataFrame(X_pca.get())
        pyreadr.write_rds(save_dir + "global_emb_pca.rds", global_emb_pca)
    

        
      

    
    
# extract fused feature
# @torch.inference_mode()
# def extract_dense_feature_map(
#     model: torch.nn.Module,
#     image: Image.Image,
#     mask: np.ndarray,
#     device: torch.device,
#     batch_size: int = 128,
#     alpha = 0.5,
#     patch_size: int = 224,
#     stride: int = 112,
#     token_size: int = 16,
#     num_workers: int = 4
# ) -> np.ndarray:
#     model.eval()
#     dataset = SlidingWindowDataset(image,mask,patch_size=patch_size, stride=stride)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=num_workers, pin_memory=True)
#     print("Total number of 224*224*3 image tiles:", len(dataset))
#     #pdb.set_trace()
#     W_pad, H_pad = dataset.image.size
#     print(f"Padded image shape:({H_pad},{W_pad})")
#     H_tokens = H_pad // token_size
#     W_tokens = W_pad // token_size
#     print(f"Padded token size:({H_tokens},{W_tokens})")
    
#     # assume D = 1024
#     D = 1024
#     D_fused = D  # final dimension：don't concatinate

#     # Initialize feature maps
#     feature_map = torch.zeros((H_tokens, W_tokens, D_fused), dtype=torch.float16, device="cpu")
#     #count_map = torch.zeros((H_tokens, W_tokens), dtype=torch.float16, device="cpu")

#     # Center weight matrix: center weight for each patch in a sliding window
#     center_weight = torch.tensor(get_center_weights(size=14), dtype=torch.float32, device="cpu")  # shape [14,14]
#     spatial_weight = center_weight  # 

#     # Local-global blending ratio (fixed)
#     alpha = 0.5 
#     print(f"alpha ={alpha}")

#     for batch, tops, lefts in tqdm(loader, desc="Sliding Window Inference"):
#         batch = batch.to(device, non_blocking=True)
#         cls_tokens, local_tokens = extract_features(model, batch)  # cls_tokens: [B,D], local_tokens: [B,196,D]
#         #pdb.set_trace()
#         # Expand shapes
#         global_expanded = cls_tokens[:, None, None, :].expand(-1, 14, 14, -1)  # [B,14,14,D]
#         local_reshaped = local_tokens.view(-1, 14, 14, D)                      # [B,14,14,D]

#         # Step 1: Fixed local-global blending (no spatial weight)
#         fused_tile = alpha * local_reshaped.cpu() + (1 - alpha) * global_expanded.cpu()  # [B,14,14,D]

#         for i in range(batch.shape[0]):
#             top_idx = tops[i].item() // token_size
#             left_idx = lefts[i].item() // token_size

#             # Step 2: Cross-window weighted accumulation
#             weighted = fused_tile[i] * spatial_weight[..., None]  # [14,14,D]
#             feature_map[top_idx:top_idx + 14, left_idx:left_idx + 14] += weighted
#             #count_map[top_idx:top_idx + 14, left_idx:left_idx + 14] += spatial_weight

#     # Final normalization
#     #final_feature_map = feature_map / count_map[..., None]  # shape [H_tokens, W_tokens, D]

#     # Crop to original image region (remove padding)
#     H_ori, W_ori = image.size[1], image.size[0]
#     H_out = H_ori // token_size
#     W_out = W_ori // token_size
#     start_y = (patch_size // 2) // token_size
#     start_x = (patch_size // 2) // token_size

#     final_map = feature_map[start_y:start_y + H_out, start_x:start_x + W_out]
#     return final_map.detach().cpu().numpy()  # [H_out, W_out, D_fused]

def run_full_pipeline(
    weight_dir,
    full_image,  # original_image，PIL.Image
    full_pixel_mask,
    full_grid_mask,
    device,
    patch_size=224,
    stride=112,
    token_size=16,
    batch_size=128,
    alpha = 0.5,
    num_workers=8,
    save_dir = None,
    sample = None,
    output_format = None
):
    t0 = time()
    device = torch.device(device)
    model = create_model(weight_dir)
    model = model.to(device)
    #model = torch.compile(model)
    model.eval()
    print("name of model：", next(model.parameters()).device)
    print("\n")
    t1=time()
    print(f"Create and setup model cost {int(t1-t0)}s!!!")
    
    print("✅✅✅✅✅✅✅✅✅✅✅ Running full pipeline..✅✅✅✅✅✅✅✅✅✅✅")
    t0 = time()
    extract_global_local_feature(
        model=model,
        image=full_image,
        mask=full_pixel_mask,
        device= device,
        batch_size=batch_size,
        patch_size=patch_size,
        stride=stride,
        token_size=token_size,
        num_workers=num_workers,
        save_dir = save_dir,
        sample = sample,
        output_format = output_format
    )

    t1 =time()
    print(f'extract 16_16_features cost {int(t1-t0)}s!!!')
     

# 

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
    args.image_path = get_image_filename(args.read_path + "he")
    print(f"image_path = {args.image_path}")
    if(not os.path.exists(f"{args.image_path}")):
        print("image file don't exist")
        exit(1)
        
    args.mask_path = args.read_path + f"/mask/mask-small.png"
    args.mask_pixel_path = args.read_path + f"/mask/mask.png"
    if(not os.path.exists(f"{args.mask_path}")):
        print("mask file don't exist")
        exit(1)
        
    if(not os.path.exists(f"{args.save_dir}")):
        os.makedirs(f"{args.save_dir}")
          
    img = load_image(args.image_path)
    print(f"raw image.shape = {np.array(img).shape}")
    img = Image.fromarray(img).convert("RGB")

    grid_mask = np.array(Image.open(args.mask_path)) > 0  # Convert to binary mask
    print(f"grid mask.shape ={grid_mask.shape}")
    
    pixel_mask = np.array(Image.open(args.mask_pixel_path)) > 0  # Convert to binary mask
    print(f"pixel mask.shape ={pixel_mask.shape}")
    
    t1=time()
    print(f"read image file, mask file(generated from HistoSweep) and create save dir cost {int(t1-t0)}s!!!")
    
    run_full_pipeline(
        weight_dir = args.weight_dir,
        full_image=img,
        full_pixel_mask = pixel_mask,
        full_grid_mask = grid_mask,
        device= args.device,
        patch_size=args.patch_size,
        stride=args.stride,
        token_size= args.token_size,
        batch_size=args.batch_size,
        num_workers= args.num_workers,
        save_dir = args.save_dir,
        sample = args.sample,
        output_format = args.output_format
    )
    
    #print(adata)
    
#     if args.clustering:
#         print("\n")
#         print("✅ 3. Clustering super pixel embedding....\n")

#         if args.use_cuML:
#             print("Use cuML library to calculate PCA and Kmeans...")
#             import cupy as cp
#             import cudf
#             from cuml.decomposition import PCA
#             from cuml.manifold import UMAP as cuUMAP
#             from cuml.cluster import KMeans as cuKMeans
#             import random
#             import cuml
#             random.seed(42)
#             np.random.seed(42)
#             start_pca = time()
#             X_cp = cp.asarray(adata.X)
#             pca = PCA(n_components=50,random_state=42)
#             X_cp = X_cp.astype('float32')
#             X_pca = pca.fit_transform(X_cp)
#             adata.obsm["X_pca"] = X_pca.get()
#             end_pca = time()
#             print(f"perform PCA dimension reduction costs {int(end_pca - start_pca)} s")

#             start_kmeans = time()
#             ncluster_list = [5, 10, 15, 20, 25, 30]
#             for cluster in ncluster_list:
#                 print(f"clustering with Kmeans(K={cluster})")
#                 t0 = time()
#                 kmeans_cuml = cuKMeans(init="k-means||", n_clusters=cluster, random_state=42)
#                 kmeans_cuml.fit(X_pca)
#                 cluster_labels = kmeans_cuml.labels_.get()
#                 adata.obs["kmeans_{}".format(cluster)] = cluster_labels.astype(str)
#             end_kmeans = time()
#             print(f"perform Kmeans costs {int(end_kmeans - start_kmeans)} s") 
#         else:
#             print("Use Sklearn library to calculate PCA and Kmeans...")
#             # perform PCA dimension reduction 
#             from sklearn.cluster import KMeans
#             from sklearn.decomposition import PCA
#             from sklearn.cluster import MiniBatchKMeans

#             start_pca = time()
#             ncluster_list = [5,10,15,20,25,30]
#             pca = PCA(n_components=50)
#             PCA_emb = pca.fit_transform(adata.X)
#             total_variance = pca.explained_variance_ratio_.sum()
#             print(f"total_variance_ratio: {total_variance:.4f}")
#             end_pca = time()
#             print(f"perform PCA dimension reduction costs {int(end_pca - start_pca)} s")
#             adata.obsm["X_pca"] = PCA_emb

#             # perform kmeans clusering
#             if(adata.X.shape[0]< 1e7):
#                 start_kmeans = time()
#                 for cluster in ncluster_list:
#                     print(f"clustering with Kmeans(K={cluster})")
#                     kmeans = KMeans(n_clusters=cluster, random_state=42)
#                     cluster_labels = kmeans.fit_predict(PCA_emb)
#                     adata.obs["kmeans_{}".format(cluster)] = cluster_labels.astype(str)
#                 end_kmeans = time()
#                 print(f"perform Kmeans costs {int(end_kmeans - start_kmeans)} s")
#             else:
#                 start_kmeans = time()
#                 batch_size = 1000
#                 ncluster_list = [5, 10, 15, 20, 25, 30]
#                 for cluster in ncluster_list:
#                     print(f"clustering with MiniBatchKMeans(K={cluster})")
#                     kmeans = MiniBatchKMeans(n_clusters=cluster, random_state=42, batch_size=batch_size)
#                     cluster_labels = kmeans.fit_predict(PCA_emb)
#                     adata.obs[f"kmeans_{cluster}"] = cluster_labels.astype(str)    
#                 end_kmeans = time()
#                 print(f"perform miniBatchKmeans costs {int(end_kmeans - start_kmeans)} s")
#         print(adata)
        
#     print("\n")
#     print("✅ 4. save embedding and coordinate to h5ad...")
#     # save h5ad
#     t0 = time()
#     h5ad_path = os.path.join(args.save_dir, f"uni_super_emb.h5ad")
#     adata.write(h5ad_path, compression="gzip")
#     t1 = time()
#     print(f"Saved embedding to {h5ad_path} with shape {adata.shape} cost {int(t1-t0)}s")
#     print("\n")
#     print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉 Congratulations, Pipeline completed sucessfully!!!!🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")

    
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