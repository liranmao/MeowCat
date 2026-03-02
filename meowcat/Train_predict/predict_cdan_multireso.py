#!/usr/bin/env python3
# Predict on the FULL IMAGE GRID (all tokens = all pixels) for ONE sample — MultiResolution
# -----------------------------------------------------------------------------
# Save as: predict_multi_resolution.py
#
# Usage:
#   python predict_multi_resolution.py PREFIX N_STATES SAMPLE_NAME \
#       --device cuda --tokens-per-chunk 16384 --chunks-per-batch 1
#
# What it does:
#   - Loads n_states checkpoints from PREFIX/states/00..(n-1)/model.ckpt
#   - Loads PREFIX/SAMPLE_NAME/embeddings-hist.pickle  (HxWxC)
#   - Treats EVERY PIXEL as a token (no spot disks)
#   - Runs MultiResolution models to get:
#       z_map: [H, W, D]   L2-normalized token embeddings (median across states)
#       p_map: [H, W, K]   per-token cell-type probs (mean across states)
#   - Saves:
#       PREFIX/SAMPLE_NAME/pred_fullgrid_outputs.pkl
#         dict(
#           z_map=[H,W,D] float32,
#           p_map=[H,W,K] float32,
#           ctypes=list[str],
#           token_dim=int,
#           n_states=int,
#           shape_hw=(H,W),
#           model_variant="MultiResolution",
#           embedding_l2_normalized=True
#         )
#
# COMPATIBILITY NOTE:
#   This script is designed to work with models trained by train_multi_resolution.py.
#   The output format is identical to predict_cdan35.py, so downstream analysis
#   code will work without modification.
# -----------------------------------------------------------------------------

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

# --- import from your codebase ---
from train_by_batch_cdan5_trainc_final2 import MultiResolutionModel
from utils import read_lines, load_pickle, save_pickle


def load_states(prefix: str, n_states: int, device: str = 'cuda') -> List[MultiResolutionModel]:
    """Load n_states model checkpoints from PREFIX/states/00..{n-1}/model.ckpt"""
    models = []
    for i in range(n_states):
        state_dir = os.path.join(prefix, 'states', f"{i:02d}")
        ckpt = os.path.join(state_dir, 'model.ckpt')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        model = MultiResolutionModel.load_from_checkpoint(ckpt, map_location=device)
        model.eval().to(device)
        models.append(model)
        print(f"[Loaded] {ckpt}")
    return models


def load_sample(prefix: str, sample_name: str):
    """Load sample embeddings and cell type names."""
    sample_dir = os.path.join(prefix, sample_name)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")
    ctypes = read_lines(os.path.join(sample_dir, 'anno-names.txt'))
    embs = load_pickle(os.path.join(sample_dir, 'embeddings-hist.pickle'))  # [H,W,C], np.ndarray
    if not isinstance(embs, np.ndarray) or embs.ndim != 3:
        raise RuntimeError(f"embeddings-hist.pickle must be [H,W,C], got {None if not hasattr(embs,'shape') else embs.shape}")
    H, W, C = embs.shape
    return sample_dir, ctypes, embs, H, W, C


def _l2_renorm_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize array along given axis."""
    n = np.sqrt(np.maximum((x * x).sum(axis=axis, keepdims=True), eps))
    return x / n


@torch.no_grad()
def predict_fullgrid(models: List[MultiResolutionModel],
                     embs: np.ndarray,      # [H,W,C]
                     tokens_per_chunk: int = 16384,
                     chunks_per_batch: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on full image grid (every pixel is a token).
    
    Returns:
      z_map: [H,W,D] float32   (median across states, then L2-renormalized)
      p_map: [H,W,K] float32   (mean across states)
    """
    device = next(models[0].parameters()).device
    dtype  = next(models[0].parameters()).dtype
    D = models[0].token_dim
    K = models[0].n_ctypes

    H, W, C = embs.shape
    Npix = H * W

    # Flatten to [Npix, C] in row-major order
    X = embs.reshape(Npix, C)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, copy=False)

    # Output buffers
    z_out = np.empty((Npix, D), dtype=np.float32)
    p_out = np.empty((Npix, K), dtype=np.float32)

    # Chunking along tokens (pixels)
    t_chunk = max(1, int(tokens_per_chunk))
    num_chunks = (Npix + t_chunk - 1) // t_chunk
    cpb = max(1, int(chunks_per_batch))

    print(f"[Predict] {Npix} pixels, {num_chunks} chunks of {t_chunk}, batch={cpb}")

    for base_chunk in range(0, num_chunks, cpb):
        batch_chunks = []
        chunk_slices = []
        for cidx in range(base_chunk, min(base_chunk + cpb, num_chunks)):
            start = cidx * t_chunk
            end   = min((cidx + 1) * t_chunk, Npix)
            x_flat = X[start:end]                               # [T_eff, C]
            if end - start < t_chunk:
                pad_len = t_chunk - (end - start)
                pad = np.zeros((pad_len, C), dtype=x_flat.dtype)
                x_fixed = np.concatenate([x_flat, pad], axis=0) # [t_chunk, C]
            else:
                x_fixed = x_flat
            batch_chunks.append(x_fixed[None, ...])             # [1,t_chunk,C]
            chunk_slices.append((start, end))

        x_bt_np = np.concatenate(batch_chunks, axis=0)          # [B, t_chunk, C]
        x_bt = torch.as_tensor(x_bt_np, device=device, dtype=dtype)

        # Run all states, then aggregate
        z_states = []
        p_states = []

        for model in models:
            # Use inp_to_lat for latent features (same as train_by_batch_cdan5.py)
            z_tok = model.inp_to_lat(x_bt)                         # [B,T,D]
            # Cell-type head (named ct_head_tok for compatibility)
            logits_ct_tok = model.ct_head_tok(z_tok)               # [B,T,K]
            p_tok = F.softmax(logits_ct_tok, dim=-1)               # [B,T,K]
            z_states.append(z_tok.detach().cpu().to(torch.float32).numpy())
            p_states.append(p_tok.detach().cpu().to(torch.float32).numpy())

        # Stack and aggregate across states
        z_stack = np.stack(z_states, axis=0)    # [S,B,T,D]
        p_stack = np.stack(p_states, axis=0)    # [S,B,T,K]
        z_med = np.median(z_stack, axis=0)      # [B,T,D]
        # Keep embeddings on unit sphere after median aggregation
        z_med = _l2_renorm_np(z_med, axis=-1)   # [B,T,D]
        p_mean = np.mean(p_stack, axis=0)       # [B,T,K]

        # Scatter back to output (remove padding for last chunk)
        for b, (start, end) in enumerate(chunk_slices):
            Teff = end - start
            z_out[start:end] = z_med[b, :Teff, :]
            p_out[start:end] = p_mean[b, :Teff, :]

        # Progress
        done = min(base_chunk + cpb, num_chunks)
        if done % 10 == 0 or done == num_chunks:
            print(f"  Processed {done}/{num_chunks} chunks...")

        # Cleanup
        del z_states, p_states, z_stack, p_stack, z_med, p_mean, x_bt

    # Reshape to maps
    z_map = z_out.reshape(H, W, D)
    p_map = p_out.reshape(H, W, K)
    return z_map, p_map


def save_outputs(sample_dir: str,
                 z_map: np.ndarray,
                 p_map: np.ndarray,
                 ctypes: List[str],
                 token_dim: int,
                 n_states: int,
                 shape_hw: Tuple[int, int],
                 out_pkl_name: str):
    """Save prediction outputs to pickle file."""
    out_path = os.path.join(sample_dir, out_pkl_name)
    obj = dict(
        z_map=z_map.astype(np.float32, copy=False),   # [H,W,D]
        p_map=p_map.astype(np.float32, copy=False),   # [H,W,K]
        ctypes=list(ctypes),
        token_dim=int(token_dim),
        n_states=int(n_states),
        shape_hw=tuple(shape_hw),
        model_variant="MultiResolution",               # Tag output
        embedding_l2_normalized=True,                  # Explicit flag
    )
    save_pickle(obj, out_path)
    print(f"[OK] Saved: {out_path}  (z_map {z_map.shape}, p_map {p_map.shape})")


def main():
    ap = argparse.ArgumentParser(
        description="Predict on full image grid using MultiResolution model"
    )
    ap.add_argument("prefix", type=str,
                    help="Top-level PREFIX used for training (contains states/*)")
    ap.add_argument("n_states", type=int,
                    help="How many states to load (00..n-1)")
    ap.add_argument("sample_name", type=str,
                    help="Sample folder under PREFIX (e.g., NCBI_001)")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Root directory containing sample data. Defaults to PREFIX.")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tokens-per-chunk", type=int, default=16384,
                    help="How many pixels per sequence chunk (controls T). Larger uses more memory.")
    ap.add_argument("--chunks-per-batch", type=int, default=1,
                    help="How many chunks to pack along batch dimension for a single forward.")
    ap.add_argument("--out-pkl-name", type=str,
                    default="pred_fullgrid_outputs_multires.pkl",
                    help="Name of output PKL written in the sample folder.")
    args = ap.parse_args()

    data_root = args.data_root if args.data_root else args.prefix

    print("=" * 60)
    print("MultiResolution Full-Grid Prediction")
    print("=" * 60)
    print(f"PREFIX: {args.prefix}")
    print(f"DATA_ROOT: {data_root}")
    print(f"SAMPLE: {args.sample_name}")
    print(f"N_STATES: {args.n_states}")
    print(f"DEVICE: {args.device}")
    print("=" * 60)

    # 1) Load models
    models = load_states(args.prefix, args.n_states, device=args.device)
    token_dim = models[0].token_dim
    n_ctypes = models[0].n_ctypes
    print(f"[Models] Loaded {len(models)} states, token_dim={token_dim}, n_ctypes={n_ctypes}")

    # 2) Load sample (full image features)
    sample_dir, ctypes, embs, H, W, C = load_sample(data_root, args.sample_name)
    print(f"[Sample] {args.sample_name}: embs {embs.shape} (H,W,C), K={len(ctypes)}, D={token_dim}")

    # 3) Predict over ALL pixels
    z_map, p_map = predict_fullgrid(
        models=models,
        embs=embs,
        tokens_per_chunk=args.tokens_per_chunk,
        chunks_per_batch=args.chunks_per_batch
    )

    # 4) Save
    save_outputs(
        sample_dir=sample_dir,
        z_map=z_map,
        p_map=p_map,
        ctypes=ctypes,
        token_dim=token_dim,
        n_states=args.n_states,
        shape_hw=(H, W),
        out_pkl_name=args.out_pkl_name
    )

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()