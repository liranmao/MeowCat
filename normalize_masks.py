#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def normalize_mask_to_single_channel(mask_path: Path, out_path: Path | None = None, threshold: int = 0):
    """
    Loads a mask PNG, converts to single-channel grayscale, binarizes it, and saves as 8-bit PNG (0/255).
    threshold=0 means any non-zero pixel becomes 255.
    """
    out_path = out_path or mask_path

    im = Image.open(mask_path)

    # Convert anything (RGBA, LA, RGB, P, etc.) to grayscale 8-bit
    im_gray = im.convert("L")
    arr = np.array(im_gray)

    # Binarize to 0/255
    bin_arr = (arr > threshold).astype(np.uint8) * 255
    out_im = Image.fromarray(bin_arr, mode="L")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_im.save(out_path)

    print(f"[OK] {mask_path} ({im.mode}, {np.array(im).shape}) -> {out_path} (L, {bin_arr.shape})")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Normalize mask PNGs to single-channel 0/255.")
    ap.add_argument("--base-dir", default="/project/CATCH/liran/histosweep",
                    help="Root directory containing sample subfolders with mask/ dirs.")
    ap.add_argument("--pattern", default="P*/mask",
                    help="Glob pattern relative to base-dir for mask directories.")
    args = ap.parse_args()
    base_dir = Path(args.base_dir)
    # Adjust pattern if needed
    sample_dirs = sorted(base_dir.glob(args.pattern))
    if not sample_dirs:
        raise SystemExit(f"No mask directories found under {base_dir}/P*/mask")

    for mask_dir in sample_dirs:
        # Adjust filenames if you have multiple masks
        for fname in ["mask.png", "mask-small.png"]:
            p = mask_dir / fname
            if p.exists():
                normalize_mask_to_single_channel(p, out_path=p, threshold=0)

if __name__ == "__main__":
    main()
