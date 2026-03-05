#!/usr/bin/env python3
"""Compare two embeddings-hist.pickle files to check if they are identical."""

import pickle
import numpy as np

path_a = "/project/KidneyHE/data_lung/P11_LUAD/embeddings-hist.pickle"
path_b = "/project/KidneyHE/01_meowcat_test/01_visium_only/input/VIS_P11_LUAD/embeddings-hist.pickle"

print(f"A (original): {path_a}")
print(f"B (test):     {path_b}")
print()

a = pickle.load(open(path_a, "rb"))
b = pickle.load(open(path_b, "rb"))

print(f"type   A: {type(a)}  B: {type(b)}")
print(f"shape  A: {a.shape}  B: {b.shape}")
print(f"dtype  A: {a.dtype}  B: {b.dtype}")
print()

print(f"exactly equal:  {np.array_equal(a, b)}")
print(f"allclose:       {np.allclose(a, b, equal_nan=True)}")
print(f"max abs diff:   {np.max(np.abs(a.astype(float) - b.astype(float)))}")
print()

# Check for NaNs / Infs
print(f"NaN count  A: {np.isnan(a).sum()}  B: {np.isnan(b).sum()}")
print(f"Inf count  A: {np.isinf(a).sum()}  B: {np.isinf(b).sum()}")
print(f"min/max    A: [{a.min():.6f}, {a.max():.6f}]  B: [{b.min():.6f}, {b.max():.6f}]")
