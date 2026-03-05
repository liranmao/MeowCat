#!/usr/bin/env python
"""Convert old batch_000.pkl (concatenated X, Y, D tuple) to per-domain
batch_vis_XXX_x/y/d.npy files expected by MeowCat training."""

import os, sys, pickle
import numpy as np


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <batch_000.pkl> <output_dir>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    out_dir = sys.argv[2]

    print(f"Loading {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 3:
        X, Y, D = data
    elif len(data) == 2:
        X, Y = data
        D = np.zeros(X.shape[0], dtype=np.int64)
        print("  No domain array in pkl — assigning all to domain 0")
    else:
        raise ValueError(f"Expected 2 or 3 arrays in pkl, got {len(data)}")

    print(f"  X: {X.shape} {X.dtype}  Y: {Y.shape} {Y.dtype}  D: {D.shape} {D.dtype}")

    os.makedirs(out_dir, exist_ok=True)

    domains = sorted(set(D.tolist()))
    for dom in domains:
        mask = D == dom
        np.save(os.path.join(out_dir, f'batch_vis_{dom:03d}_x.npy'), X[mask])
        np.save(os.path.join(out_dir, f'batch_vis_{dom:03d}_y.npy'), Y[mask])
        np.save(os.path.join(out_dir, f'batch_vis_{dom:03d}_d.npy'), D[mask])
        print(f"  Domain {dom}: {mask.sum()} spots")

    print(f"\nSaved {len(domains)} domain batch files to {out_dir}")


if __name__ == "__main__":
    main()
