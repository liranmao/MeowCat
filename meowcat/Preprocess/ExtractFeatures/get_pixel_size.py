#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path
from UTILS import get_image_filename


def get_args():
    parser = argparse.ArgumentParser(description="Extract microns-per-pixel from image metadata")
    parser.add_argument('--read_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--sample', type=str, default="AAAA")
    parser.add_argument('--raw_flag', type=str, default="he-raw")
    parser.add_argument('--pixel_size_raw', type=float, default=None,
                        help="Manual pixel size override (microns-per-pixel). "
                             "If provided, skip auto-detection and use this value.")
    args = parser.parse_args()
    return args


def get_microns_per_pixel_x(path: str) -> float:
    """
    Return microns-per-pixel in X for TIFF/SVS/JPG files.
    
    Tries, in order:
      1) OpenSlide metadata (openslide.mpp-x / aperio.MPP) for WSI/SVS
      2) TIFF tags XResolution + ResolutionUnit via tifffile
      3) Pillow for JPG/PNG (JFIF or EXIF metadata)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    mpp_x = None

    # 1) Try OpenSlide for WSI/SVS
    if suffix in ('.svs', '.tif', '.tiff', '.ndpi', '.vms', '.scn'):
        try:
            import openslide
            slide = openslide.OpenSlide(str(path))
            props = slide.properties

            for key in ("openslide.mpp-x", "aperio.MPP"):
                if key in props:
                    mpp_x = float(props[key])
                    break
            slide.close()
            
            if mpp_x is not None:
                return mpp_x
        except Exception:
            pass

    # 2) For TIFF: use tifffile
    if suffix in ('.tif', '.tiff') and mpp_x is None:
        try:
            from tifffile import TiffFile

            with TiffFile(str(path)) as tif:
                page = tif.pages[0]
                tags = page.tags

                x_res_tag = tags.get("XResolution")
                res_unit_tag = tags.get("ResolutionUnit")

                if x_res_tag is not None and res_unit_tag is not None:
                    x_res = x_res_tag.value
                    try:
                        num, den = x_res
                        x_dpi = num / den
                    except TypeError:
                        x_dpi = float(x_res)

                    unit_code = int(res_unit_tag.value)

                    if unit_code == 2:  # inch
                        mpp_x = 25400.0 / x_dpi
                    elif unit_code == 3:  # centimeter
                        mpp_x = 10000.0 / x_dpi
                    
                    if mpp_x is not None:
                        return mpp_x
        except Exception:
            pass

    # 3) For JPG/PNG: use Pillow
    if suffix in ('.jpg', '.jpeg', '.png'):
        try:
            from PIL import Image

            with Image.open(str(path)) as img:
                # Check JFIF/basic info
                dpi = img.info.get('dpi')
                
                if dpi is None:
                    # Try EXIF
                    exif = img.getexif()
                    if exif:
                        # EXIF tags: 282=XResolution, 283=YResolution, 296=ResolutionUnit
                        x_res = exif.get(282)
                        res_unit = exif.get(296, 2)  # default to inches
                        
                        if x_res is not None:
                            if hasattr(x_res, 'numerator'):
                                x_dpi = x_res.numerator / x_res.denominator
                            else:
                                x_dpi = float(x_res)
                            
                            if res_unit == 2:  # inches
                                mpp_x = 25400.0 / x_dpi
                            elif res_unit == 3:  # centimeters
                                mpp_x = 10000.0 / x_dpi
                else:
                    x_dpi = dpi[0]
                    mpp_x = 25400.0 / x_dpi  # assumes DPI (inches)
                    
        except Exception:
            pass

    if mpp_x is None:
        raise RuntimeError(f"Could not extract resolution metadata from {path}")
    
    return mpp_x


def main():
    args = get_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    output_path = os.path.join(args.save_dir, "pixel-size-raw.txt")

    # If pixel-size-raw.txt already exists (user-provided per-sample override), use it as-is
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_val = float(f.read().strip())
        print(f"Using existing pixel-size-raw.txt: {existing_val:.10f} (auto-detection skipped)")
        return

    # If manual pixel size is provided, skip auto-detection
    if args.pixel_size_raw is not None:
        mpp_x = args.pixel_size_raw
        print(f"Using manual pixel size: {mpp_x:.10f} (auto-detection skipped)")
        with open(output_path, 'w') as f:
            f.write(f"{mpp_x:.10f}\n")
        print(f"Saved to: {output_path}")
        return

    # Auto-detect from image metadata
    args.image_path = get_image_filename(args.read_dir + args.raw_flag)
    print(f"Image path: {args.image_path}")

    if not os.path.exists(args.image_path):
        print("Image file doesn't exist")
        sys.exit(1)

    try:
        mpp_x = get_microns_per_pixel_x(args.image_path)
        print(f"Microns per pixel (X): {mpp_x:.10f}")

        with open(output_path, 'w') as f:
            f.write(f"{mpp_x:.10f}\n")
        print(f"Saved to: {output_path}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    from time import time
    t0 = time()
    main()
    t1 = time()
    print(f"Running done, cost {t1-t0:.2f}s")