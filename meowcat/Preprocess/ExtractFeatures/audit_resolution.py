#!/usr/bin/env python
"""
Audit image resolutions across all samples.
Checks for:
1. Resolution consistency across images
2. Images with resolution < target (would need upscaling)
3. Images where resolution cannot be read
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# Import your existing function
from UTILS import get_image_filename


def get_args():
    parser = argparse.ArgumentParser(description="Audit image resolutions across samples")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory containing sample folders")
    parser.add_argument('--pattern', type=str, default="GBM*", help="Folder pattern to match (default: GBM*)")
    parser.add_argument('--raw_flag', type=str, default="he_raw", help="Raw flag for image filename")
    parser.add_argument('--target_mpp', type=float, default=0.5, help="Target microns per pixel (default: 0.5)")
    args = parser.parse_args()
    return args


def get_microns_per_pixel_x(path: str) -> tuple[float | None, str]:
    """
    Return (mpp_x, method) or (None, error_message) for TIFF/SVS/JPG files.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    mpp_x = None
    method = None

    # 1) Try OpenSlide for WSI/SVS
    if suffix in ('.svs', '.tif', '.tiff', '.ndpi', '.vms', '.scn'):
        try:
            import openslide
            slide = openslide.OpenSlide(str(path))
            props = slide.properties

            for key in ("openslide.mpp-x", "aperio.MPP"):
                if key in props:
                    mpp_x = float(props[key])
                    method = f"OpenSlide ({key})"
                    break
            slide.close()
            
            if mpp_x is not None:
                return mpp_x, method
        except Exception as e:
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
                        method = "tifffile (inches)"
                    elif unit_code == 3:  # centimeter
                        mpp_x = 10000.0 / x_dpi
                        method = "tifffile (cm)"
                    
                    if mpp_x is not None:
                        return mpp_x, method
                else:
                    return None, "TIFF missing XResolution or ResolutionUnit tags"
        except Exception as e:
            return None, f"tifffile error: {e}"

    # 3) For JPG/PNG: use Pillow
    if suffix in ('.jpg', '.jpeg', '.png'):
        try:
            from PIL import Image

            with Image.open(str(path)) as img:
                dpi = img.info.get('dpi')
                
                if dpi is None:
                    exif = img.getexif()
                    if exif:
                        x_res = exif.get(282)
                        res_unit = exif.get(296, 2)
                        
                        if x_res is not None:
                            if hasattr(x_res, 'numerator'):
                                x_dpi = x_res.numerator / x_res.denominator
                            else:
                                x_dpi = float(x_res)
                            
                            # Check for fake/default DPI values
                            if x_dpi in (72, 96, 150):
                                return None, f"JPG has likely default DPI ({x_dpi}) - not reliable"
                            
                            if res_unit == 2:
                                mpp_x = 25400.0 / x_dpi
                                method = "Pillow EXIF (inches)"
                            elif res_unit == 3:
                                mpp_x = 10000.0 / x_dpi
                                method = "Pillow EXIF (cm)"
                            
                            if mpp_x is not None:
                                return mpp_x, method
                        else:
                            return None, "JPG has no XResolution in EXIF"
                    else:
                        return None, "JPG has no EXIF metadata"
                else:
                    x_dpi = dpi[0]
                    
                    # Check for fake/default DPI values
                    if x_dpi in (72, 96, 150):
                        return None, f"JPG has likely default DPI ({x_dpi}) - not reliable"
                    
                    mpp_x = 25400.0 / x_dpi
                    method = "Pillow JFIF"
                    return mpp_x, method
                    
        except Exception as e:
            return None, f"Pillow error: {e}"

    return None, f"Unsupported format or no metadata found"


def main():
    args = get_args()
    
    import glob
    
    # Find all matching folders
    pattern = os.path.join(args.base_dir, args.pattern, "")
    sample_dirs = sorted(glob.glob(pattern))
    
    if not sample_dirs:
        print(f"No folders found matching {pattern}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"RESOLUTION AUDIT")
    print(f"Base directory: {args.base_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Target MPP: {args.target_mpp}")
    print(f"Found {len(sample_dirs)} sample folders")
    print("=" * 80)
    
    # Results storage
    results = []
    successful = []
    failed = []
    needs_upscale = []
    
    for sample_path in sample_dirs:
        sample = os.path.basename(sample_path.rstrip('/'))
        feature_dir = sample_path
        
        # Get image path
        try:
            image_path = get_image_filename(os.path.join(feature_dir, args.raw_flag))
        except Exception as e:
            failed.append({
                'sample': sample,
                'error': f"get_image_filename failed: {e}",
                'image_path': None
            })
            continue
        
        if not os.path.exists(image_path):
            failed.append({
                'sample': sample,
                'error': "Image file not found",
                'image_path': image_path
            })
            continue
        
        # Get resolution
        mpp_x, method_or_error = get_microns_per_pixel_x(image_path)
        
        ext = Path(image_path).suffix.lower()
        
        if mpp_x is None:
            failed.append({
                'sample': sample,
                'error': method_or_error,
                'image_path': image_path,
                'extension': ext
            })
        else:
            scale = mpp_x / args.target_mpp
            result = {
                'sample': sample,
                'mpp': mpp_x,
                'method': method_or_error,
                'image_path': image_path,
                'extension': ext,
                'scale': scale
            }
            successful.append(result)
            
            if scale < 1.0:
                needs_upscale.append(result)
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("SUCCESSFUL READS")
    print("=" * 80)
    
    if successful:
        # Group by MPP value
        mpp_groups = defaultdict(list)
        for r in successful:
            mpp_rounded = round(r['mpp'], 4)
            mpp_groups[mpp_rounded].append(r['sample'])
        
        for r in successful:
            status = "⚠️ UPSCALE" if r['scale'] < 1.0 else "✓"
            print(f"{status} {r['sample']:<25} | MPP: {r['mpp']:.6f} | Scale: {r['scale']:.4f} | {r['extension']} | {r['method']}")
    else:
        print("No successful reads!")
    
    print("\n" + "=" * 80)
    print("FAILED READS")
    print("=" * 80)
    
    if failed:
        for r in failed:
            ext = r.get('extension', 'N/A')
            print(f"✗ {r['sample']:<25} | {ext} | Error: {r['error']}")
    else:
        print("All images read successfully!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"Total samples:        {len(sample_dirs)}")
    print(f"Successful reads:     {len(successful)}")
    print(f"Failed reads:         {len(failed)}")
    
    if successful:
        mpps = [r['mpp'] for r in successful]
        print(f"\nResolution range:     {min(mpps):.6f} - {max(mpps):.6f} MPP")
        
        unique_mpps = set(round(m, 4) for m in mpps)
        if len(unique_mpps) == 1:
            print(f"Resolution consistency: ✓ All same ({list(unique_mpps)[0]} MPP)")
        else:
            print(f"Resolution consistency: ⚠️ MULTIPLE RESOLUTIONS FOUND")
            mpp_groups = defaultdict(list)
            for r in successful:
                mpp_groups[round(r['mpp'], 4)].append(r['sample'])
            for mpp, samples in sorted(mpp_groups.items()):
                print(f"  - {mpp} MPP: {len(samples)} samples")
    
    print(f"\nNeeds upscaling (mpp < {args.target_mpp}): {len(needs_upscale)}")
    if needs_upscale:
        for r in needs_upscale:
            print(f"  - {r['sample']}: {r['mpp']:.6f} MPP (scale would be {r['scale']:.4f}x)")
    
    if failed:
        print(f"\nFailed by extension:")
        ext_groups = defaultdict(list)
        for r in failed:
            ext_groups[r.get('extension', 'unknown')].append(r['sample'])
        for ext, samples in sorted(ext_groups.items()):
            print(f"  - {ext}: {len(samples)} ({', '.join(samples)})")
    
    # Exit code
    if failed or needs_upscale:
        print("\n⚠️ Issues detected - review before proceeding!")
        sys.exit(1)
    else:
        print("\n✓ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()