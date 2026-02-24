import sys 
sys.path.append("/project/KidneyHE/xiaokang_result/UNI_V7.1/ExtractFeatures/HistoSweep-main/HistoSweep/")
import shutil
import argparse
import pandas as pd
import numpy as np
import os 
from time import time
import pdb
from saveParameters import saveParams
from computeMetrics import compute_metrics_memory_optimized
from densityFiltering import compute_low_density_mask
from textureAnalysis import run_texture_analysis
from ratioFiltering import run_ratio_filtering
from generateMask import generate_final_mask
from additionalPlots import generate_additionalPlots
from PIL import Image

from UTILS import get_image_filename,load_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, default='AAAA',
                       help='dictionary to read dataset')
    parser.add_argument('--save_dir', type=str, default='BBBB',
                       help='Directory to save results')
    parser.add_argument('--pixel_size_raw',type=float,default = 0.5)
    parser.add_argument('--density_thresh',type=int,default = 100)
    parser.add_argument('--clean_background_flag', action='store_true', help='Wheter to preserve fibrous regions that are otherwise being incorrectly filtered out')
    parser.add_argument('--min_size',type=int,default = 10)
    parser.add_argument('--patch_size',type=int,default = 16)
    parser.add_argument('--pixel_size',type=float,default = 0.5)
    
    ##########################
    return parser.parse_args()
    
    
def main():
    #pdb.set_trace()
    args = get_args()
    print(args)
    # Flag for whether to rescale the image 
    need_scaling_flag = False  # True if image resolution ≠ 0.5µm (or desired size) per pixel
    # Flag for whether to preprocess the image 
    need_preprocessing_flag = False  # True if image dimensions are not divisible by patch_size
    HE_prefix = args.read_dir
    directory = args.save_dir
    pixel_size_raw = args.pixel_size_raw
    density_thresh = args.density_thresh
    clean_background_flag = args.clean_background_flag
    min_size = args.min_size
    patch_size = args.patch_size
    pixel_size = args.pixel_size
    
    
    # Read dataset
    if not os.path.exists(args.read_dir):
        raise ValueError(f"Path file {args.read_dir} does not exist!")
    
    # Create save directory
    #os.makedirs(args.save_dir, exist_ok=True)
    

    ########################main code###################################
    # rescale and preprocess image
    #image = load_image(os.path.join(HE_prefix, "he_processed.png"))
    image = load_image(get_image_filename(os.path.join(HE_prefix, "he")))
    print(image.shape)
    
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#########################################################################################
#     saveParams(HE_prefix, need_scaling_flag, need_preprocessing_flag, pixel_size_raw,density_thresh,clean_background_flag,min_size,patch_size,pixel_size)
#########################################################################################

    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics_memory_optimized(image, patch_size=patch_size)
    
    # identify low density superpixels
    mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
    
    print('Total selected for density filtering: ', mask1_lowdensity.sum())
    
    # perform texture analysis 
#     mask1_lowdensity_update = run_texture_analysis(prefix=HE_prefix, image=image, tissue_mask=mask1_lowdensity, patch_size=patch_size, glcm_levels=64)
    # perform texture analysis 
    mask1_lowdensity_update = run_texture_analysis(prefix=HE_prefix, image=image, tissue_mask=mask1_lowdensity, output_dir=directory, patch_size=patch_size, glcm_levels=64)

    
    # identify low ratio superpixels
    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
    print(mask2_lowratio.shape)
    
    
    if not os.path.exists(os.path.join(f"{HE_prefix}/{directory}")):
        os.makedirs(os.path.join(f"{HE_prefix}/{directory}"))
    generate_final_mask(prefix=HE_prefix, he=image,output_dir=directory, 
                    mask1_updated = mask1_lowdensity_update, mask2 = mask2_lowratio, 
                    clean_background = clean_background_flag, 
                    super_pixel_size=patch_size, minSize = min_size)

    ###########################################################
    
    print("Running successfully!")
    # don't copy mask file
#     print("copy mask-small.png to its parent folder...")
#     file_path = os.path.join(f"{HE_prefix}/{directory}", 'mask-small.png')
#     dest_path = os.path.join(f"{HE_prefix}/", 'mask-small.png')
#     shutil.copy2(file_path, dest_path) 
#     print(f"File copied to {dest_path}")
    
#     print("copy mask.png to its parent folder...")
#     file_path = os.path.join(f"{HE_prefix}/{directory}", 'mask.png')
#     dest_path = os.path.join(f"{HE_prefix}/", 'mask.png')
#     shutil.copy2(file_path, dest_path) 
#     print(f"File copied to {dest_path}")
    
#     print("Copying successfully!!!!")

if __name__ == '__main__':
    t0 = time()
    main()
    t1 = time()
    print(f"Running this file cost {t1-t0} s!!!")