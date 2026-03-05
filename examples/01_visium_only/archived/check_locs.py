#!/usr/bin/env python3
"""Compare locs.tsv between original and test directories by barcode."""

import pandas as pd
import numpy as np

path_a = "/project/KidneyHE/data_lung/P11_LUAD/locs.tsv"
path_b = "/project/KidneyHE/01_meowcat_test/01_visium_only/input/VIS_P11_LUAD/locs.tsv"

print(f"A (original): {path_a}")
print(f"B (test):     {path_b}")
print()

a = pd.read_csv(path_a, sep="\t").set_index("spot")
b = pd.read_csv(path_b, sep="\t").set_index("spot")

print(f"shape  A: {a.shape}  B: {b.shape}")
print()

# Barcode overlap
common = a.index.intersection(b.index)
only_a = a.index.difference(b.index)
only_b = b.index.difference(a.index)
print(f"common barcodes: {len(common)}")
print(f"only in A: {len(only_a)}")
print(f"only in B: {len(only_b)}")

if len(only_a) > 0:
    print(f"  A-only examples: {list(only_a[:5])}")
if len(only_b) > 0:
    print(f"  B-only examples: {list(only_b[:5])}")
print()

# Compare x, y for common barcodes
ac = a.loc[common]
bc = b.loc[common]

diff_x = (ac["x"] - bc["x"]).abs()
diff_y = (ac["y"] - bc["y"]).abs()

print(f"x: max_diff={diff_x.max()}  mean_diff={diff_x.mean():.4f}  nonzero={(diff_x > 0).sum()}")
print(f"y: max_diff={diff_y.max()}  mean_diff={diff_y.mean():.4f}  nonzero={(diff_y > 0).sum()}")
print(f"exactly equal (x and y): {(diff_x == 0).all() and (diff_y == 0).all()}")
print()

# Show a few examples of mismatches if any
mismatch = (diff_x > 0) | (diff_y > 0)
n_mismatch = mismatch.sum()
print(f"mismatched spots: {n_mismatch} / {len(common)}")
if n_mismatch > 0:
    examples = ac[mismatch].head(10).copy()
    examples.columns = ["A_x", "A_y"]
    examples["B_x"] = bc.loc[examples.index, "x"]
    examples["B_y"] = bc.loc[examples.index, "y"]
    examples["dx"] = examples["A_x"] - examples["B_x"]
    examples["dy"] = examples["A_y"] - examples["B_y"]
    print("\nSample mismatches:")
    print(examples.to_string())

    # Check if there's a systematic pattern (swap, flip, scale)
    print("\n--- Checking systematic patterns on all mismatches ---")
    ax = ac.loc[mismatch, "x"].values.astype(float)
    ay = ac.loc[mismatch, "y"].values.astype(float)
    bx = bc.loc[mismatch, "x"].values.astype(float)
    by = bc.loc[mismatch, "y"].values.astype(float)

    # Check: is B swapped? (B_x ~ A_y, B_y ~ A_x)
    corr_xx = np.corrcoef(ax, bx)[0, 1]
    corr_xy = np.corrcoef(ax, by)[0, 1]
    corr_yx = np.corrcoef(ay, bx)[0, 1]
    corr_yy = np.corrcoef(ay, by)[0, 1]
    print(f"corr(A_x, B_x)={corr_xx:.4f}  corr(A_x, B_y)={corr_xy:.4f}")
    print(f"corr(A_y, B_x)={corr_yx:.4f}  corr(A_y, B_y)={corr_yy:.4f}")

    # Check scale ratio
    if np.std(bx) > 0 and np.std(by) > 0:
        ratio_xx = np.median(ax / bx) if np.all(bx != 0) else float("nan")
        ratio_yy = np.median(ay / by) if np.all(by != 0) else float("nan")
        ratio_xy = np.median(ax / by) if np.all(by != 0) else float("nan")
        ratio_yx = np.median(ay / bx) if np.all(bx != 0) else float("nan")
        print(f"median ratio A_x/B_x={ratio_xx:.4f}  A_y/B_y={ratio_yy:.4f}")
        print(f"median ratio A_x/B_y={ratio_xy:.4f}  A_y/B_x={ratio_yx:.4f}")
