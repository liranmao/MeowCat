import argparse
import os 
import numpy as np
from time import time
from PIL import Image
import pdb 
#### Load package ####
import PIL
import tifffile
import cv2
import os
from UTILS import get_image_filename,load_image,rescale_image_cv2,adjust_margins,smart_save_image
Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = 10e100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--scale_value',type = float, default = 1.0)
    parser.add_argument('--pad',type = int,default = 16)
    parser.add_argument('--sample',type=str,default="AAAA")
    parser.add_argument('--raw_flag',type=str,default="he-raw")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(args)
    #pdb.set_trace()
    args.image_path =get_image_filename(args.read_dir + args.raw_flag)
    print(args.image_path)
    if(not os.path.exists(f"{args.image_path}")):
        print("image file don't exist")
        exit(1)
        
    if(not os.path.exists(f"{args.save_dir}")):
        os.makedirs(f"{args.save_dir}")
    print(args.save_dir)
        
    img = load_image(args.image_path)
    print(f"image_raw.shape ={img.shape}")
         
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {args.scale_value:.3f})...')
    t0 = time()
    img = rescale_image_cv2(img, args.scale_value)
    print(int(time() - t0), 'sec')
    img = img.astype(np.uint8)
    print(img.shape)
    img = adjust_margins(img, pad=args.pad, pad_value=255)
    
    smart_save_image(img, args.save_dir, base_name="he", size_threshold=50000)
### save file in he.jpg or he.tiff
###         
if __name__ == '__main__':
    t0 = time()
    main()
    t1 = time()
    print(f"running done,cost {t1-t0}s")
