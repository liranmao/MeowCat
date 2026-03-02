#!/usr/bin/env python3
"""
trainc_final.py
Enhanced Multi-Resolution Training: Visium (soft labels, MSE) + Xenium (hard labels, CE)
Shared encoder with resolution-aware loss computation.

NEW FEATURES:
  1. Parallel training of multiple states (--parallel-states)
  2. Sequential training mode: Visium first, then Xenium (--sequential-training)
  3. 3-Phase training: Combine --two-stage + --sequential-training for:
     Phase 0: Reconstruction pretraining (--epochs1)
     Phase 1: Visium training (--visium-epochs)
     Phase 2: Xenium fine-tuning (--xenium-epochs)

Supports:
  - Two-stage training (stage1: recon only, stage2: weak + CDAN)
  - Multiple model states (n-states)
  - CDAN with entropy conditioning
  - OOS (out-of-sample) monitoring
  - Encoder freezing
  - [NEW] Parallel state training across GPUs
  - [NEW] Sequential Visium->Xenium training
  - [NEW] 3-Phase training (Recon -> Visium -> Xenium)

Usage examples:
    # Standard training (joint Visium + Xenium)
    python train_multi_resolution_enhanced.py /path/to/batches \
        --two-stage --epochs1 15 --epochs2 200 \
        --n-states 2 --adv-lambda 0.005

    # Parallel state training (train all states simultaneously)
    python train_multi_resolution_enhanced.py /path/to/batches \
        --two-stage --epochs1 15 --epochs2 200 \
        --n-states 4 --parallel-states --gpus 0,1,2,3

    # Sequential training: Visium first, then Xenium
    python train_multi_resolution_enhanced.py /path/to/batches \
        --sequential-training --visium-epochs 100 --xenium-epochs 50 \
        --n-states 2

    # 3-PHASE TRAINING: Recon -> Visium -> Xenium (RECOMMENDED)
    python train_multi_resolution_enhanced.py /path/to/batches \
        --two-stage --epochs1 15 \
        --sequential-training --visium-epochs 100 --xenium-epochs 50 \
        --n-states 2 --freeze-encoder-n 2 --recon-weight 0.1
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.optim import Adam
import gc
import math
from time import time
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Import from your existing code
from train_by_batch_cdan5 import (
    FeedForward,
    GradReverse,
    MetricTracker,
    auto_batch_size,
    save_pickle,
    load_pickle,
    prepare_and_save_batches,
    get_disk_mask,
    _read_radius,
    create_train_val_split,
)

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.progress import RichProgressBar

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_FACTOR = 4

# Technology name -> resolution ID mapping
TECH_TO_RESOLUTION = {
    'vis': 0,      # Visium -> resolution 0 (MSE loss)
    'xen': 1,      # Xenium -> resolution 1 (CE loss)
}


# ---------- Dataset with Resolution Tracking ----------
class MultiResolutionDataset(Dataset):
    """
    Dataset that tracks resolution type per sample.
    Expects batches named: batch_XXX_x.npy, batch_XXX_y.npy, batch_XXX_d.npy
    
    Resolution mapping (customize as needed):
        batch_000_* -> resolution 0 (Visium)
        batch_001_* -> resolution 1 (Xenium)
    """
    
    def __init__(self, batch_dir: str, resolution_map: dict = None, 
                tech_to_resolution: dict = None, max_cached_batches: int = 4,
                filter_resolution: int = None):
        """
        Args:
            batch_dir: Directory containing batch_*_x.npy files
            resolution_map: Manual override: {batch_stem: resolution_id}
            tech_to_resolution: Tech name -> resolution mapping
                            e.g., {'vis': 0, 'xen': 1}
                            If None, uses default: vis=0, xen=1
            filter_resolution: If set, only include samples with this resolution
                              (0 for Visium only, 1 for Xenium only, None for all)
        """
        super().__init__()
        self.batch_dir = batch_dir
        self.max_cached_batches = max_cached_batches
        self.filter_resolution = filter_resolution
        
        # Default tech -> resolution mapping
        if tech_to_resolution is None:
            tech_to_resolution = {
                'vis': 0,
                'xen': 1,
            }
        self.tech_to_resolution = tech_to_resolution
        
        # Find all batch stems (supports both batch_XXX and batch_tech_XXX formats)
        npy_x = sorted(f for f in os.listdir(batch_dir) 
                    if f.startswith("batch_") and f.endswith("_x.npy"))
        self.batch_stems = [f[:-6] for f in npy_x]  # e.g., "batch_vis_000"
        
        if not self.batch_stems:
            raise RuntimeError(f"No batch files found in {batch_dir}")
        
        # Auto-detect resolution from batch stem
        if resolution_map is None:
            resolution_map = {}
            for stem in self.batch_stems:
                parts = stem.split("_")
                
                if len(parts) == 3:
                    tech = parts[1]
                    if tech in tech_to_resolution:
                        resolution_map[stem] = tech_to_resolution[tech]
                    else:
                        print(f"  Warning: Unknown tech '{tech}' in {stem}, defaulting to resolution 0")
                        resolution_map[stem] = 0
                elif len(parts) == 2:
                    idx = int(parts[1])
                    resolution_map[stem] = idx
                else:
                    print(f"  Warning: Unexpected batch format {stem}, defaulting to resolution 0")
                    resolution_map[stem] = 0
        
        self.resolution_map = resolution_map
        
        # Build index: (batch_idx, local_idx, resolution)
        self.samples = []
        self.n_inp = None
        self.n_out = None
        self.token_len = None
        self.n_domains = 0
        self.has_domain = False

        for batch_idx, stem in enumerate(self.batch_stems):
            x_path = os.path.join(batch_dir, stem + "_x.npy")
            y_path = os.path.join(batch_dir, stem + "_y.npy")
            d_path = os.path.join(batch_dir, stem + "_d.npy")
            
            x = np.load(x_path, mmap_mode='r')
            y = np.load(y_path, mmap_mode='r')
            
            n_samples = x.shape[0]
            if self.n_inp is None:
                self.token_len = x.shape[1]
                self.n_inp = x.shape[2]
                self.n_out = y.shape[1]
            
            if os.path.exists(d_path):
                d = np.load(d_path, mmap_mode='r')
                if d.size > 0 and d.max() >= 0:
                    self.has_domain = True
                    self.n_domains = max(self.n_domains, int(d.max()) + 1)
                del d
            
            resolution = self.resolution_map.get(stem, 0)
            
            # Filter by resolution if specified
            if filter_resolution is not None and resolution != filter_resolution:
                continue
        
            for local_idx in range(n_samples):
                self.samples.append((batch_idx, local_idx, resolution))
            
            del x, y
        
        # Print summary
        filter_str = f" (filtered to resolution={filter_resolution})" if filter_resolution is not None else ""
        print(f"MultiResolutionDataset: {len(self.samples)} samples{filter_str}")
        print(f"  T={self.token_len}, C={self.n_inp}, K={self.n_out}")
        
        res_counts = {}
        for _, _, r in self.samples:
            res_counts[r] = res_counts.get(r, 0) + 1
        
        res_to_tech = {v: k for k, v in tech_to_resolution.items()}
        for r, c in sorted(res_counts.items()):
            tech_name = res_to_tech.get(r, f"res_{r}")
            print(f"  {tech_name} (resolution={r}): {c} samples")
        print(f"  n_domains={self.n_domains}, has_domain={self.has_domain}")
        
        self._cache = {}
        self._cache_order = []
    
    def _load_batch(self, batch_idx):
        if batch_idx in self._cache:
            return self._cache[batch_idx]
        
        stem = self.batch_stems[batch_idx]
        x = np.load(os.path.join(self.batch_dir, stem + "_x.npy"), mmap_mode='r')
        y = np.load(os.path.join(self.batch_dir, stem + "_y.npy"), mmap_mode='r')
        
        d_path = os.path.join(self.batch_dir, stem + "_d.npy")
        d = np.load(d_path, mmap_mode='r') if os.path.exists(d_path) else None
        
        self._cache[batch_idx] = (x, y, d)
        self._cache_order.append(batch_idx)
        
        while len(self._cache) > self.max_cached_batches:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        
        return self._cache[batch_idx]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        batch_idx, local_idx, resolution = self.samples[idx]
        x, y, d = self._load_batch(batch_idx)
        
        x_sample = np.array(x[local_idx], dtype=np.float32)
        y_sample = np.array(y[local_idx], dtype=np.float32)
        d_sample = int(d[local_idx]) if d is not None else -1
        
        return x_sample, y_sample, d_sample, resolution
    
    def close(self):
        self._cache.clear()
        self._cache_order.clear()
        gc.collect()


class FilteredResolutionDataset(Dataset):
    """
    Wrapper that filters an existing MultiResolutionDataset by resolution.
    Useful for sequential training (Visium first, then Xenium).
    """
    def __init__(self, base_dataset: MultiResolutionDataset, resolution: int):
        """
        Args:
            base_dataset: The full MultiResolutionDataset
            resolution: Only include samples with this resolution (0=Visium, 1=Xenium)
        """
        self.base_dataset = base_dataset
        self.resolution = resolution
        
        # Build filtered index
        self.filtered_indices = [
            i for i, (_, _, r) in enumerate(base_dataset.samples)
            if r == resolution
        ]
        
        # Copy attributes
        self.n_inp = base_dataset.n_inp
        self.n_out = base_dataset.n_out
        self.token_len = base_dataset.token_len
        self.n_domains = base_dataset.n_domains
        self.has_domain = base_dataset.has_domain
        
        print(f"FilteredResolutionDataset: {len(self.filtered_indices)} samples "
              f"(resolution={resolution})")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.filtered_indices[idx]]


def collate_fn(batch):
    """Custom collate that groups by resolution for efficient loss computation."""
    x_list, y_list, d_list, r_list = zip(*batch)
    
    x = torch.tensor(np.stack(x_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    d = torch.tensor(d_list, dtype=torch.long)
    r = torch.tensor(r_list, dtype=torch.long)
    
    return x, y, d, r


# ---------- OOS Monitor Callback ----------
class OOSMonitorMultiRes(pl.Callback):
    """Out-of-sample monitor for multi-resolution model."""
    def __init__(self, sample_dir: str, prefix: str,
                 tmp_dir: str = None, radius: int = None,
                 device: str = 'cuda', batch_size: int = 64,
                 resolution: int = 0):
        super().__init__()
        self.sample_dir = sample_dir
        self.prefix = prefix
        self.tmp_dir = tmp_dir
        self.radius = radius
        self.device = device
        self.batch_size = batch_size
        self.resolution = resolution
        self.records = []
        
        self.out_tsv = os.path.join(prefix, "oos_loss_curve.tsv")
        self.out_png = os.path.join(prefix, "oos_loss_curve.png")
    
    def setup(self, trainer, pl_module, stage: str):
        if self.tmp_dir is None:
            self.tmp_dir = os.path.join(self.sample_dir, "_oos_batch")
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        have_batches = any(
            f.startswith("batch_") and f.endswith("_x.npy")
            for f in os.listdir(self.tmp_dir)
        ) if os.path.exists(self.tmp_dir) else False
        
        if not have_batches:
            r = self.radius if self.radius else _read_radius(self.sample_dir)
            if r is None:
                raise RuntimeError(f"Could not read radius from {self.sample_dir}")
            prepare_and_save_batches([self.sample_dir], r, self.tmp_dir, 
                                    samples_per_batch=1, domain_map_file=None)
        
        self._oos_dataset = MultiResolutionDataset(
            self.tmp_dir, 
            resolution_map={0: self.resolution}
        )
        self._oos_loader = DataLoader(
            self._oos_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False,
        )
    
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        device = self.device
        dtype = next(pl_module.parameters()).dtype
        
        total_N = 0
        sum_loss = sum_weak = sum_recon = 0.0
        
        for x, y, d, r in self._oos_loader:
            B = x.shape[0]
            total_N += B
            
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            r = r.to(device=device)
            
            losses = pl_module.compute_loss(x, y, d.to(device), r)
            
            sum_loss += float(losses['loss'].item()) * B
            sum_weak += float(losses['loss_visium'].item() + losses['loss_xenium'].item()) * B
            sum_recon += float(losses['loss_recon'].item()) * B
        
        if total_N > 0:
            oos_loss = sum_loss / total_N
            oos_weak = sum_weak / total_N
            oos_recon = sum_recon / total_N
            
            pl_module.log('oos_val_loss', oos_loss, prog_bar=True)
            pl_module.log('oos_weak_mse', oos_weak)
            pl_module.log('oos_recon', oos_recon)
            
            self.records.append({
                'epoch': trainer.current_epoch,
                'oos_loss': oos_loss,
                'oos_weak': oos_weak,
                'oos_recon': oos_recon,
                'n_samples': total_N,
            })
    
    def on_fit_end(self, trainer, pl_module):
        os.makedirs(self.prefix, exist_ok=True)
        
        with open(self.out_tsv, 'w') as f:
            f.write("epoch\toos_loss\toos_weak\toos_recon\tn_samples\n")
            for r in self.records:
                f.write(f"{r['epoch']}\t{r['oos_loss']:.6f}\t{r['oos_weak']:.6f}\t"
                       f"{r['oos_recon']:.6f}\t{r['n_samples']}\n")
        
        try:
            import matplotlib.pyplot as plt
            epochs = [r['epoch'] for r in self.records]
            losses = [r['oos_loss'] for r in self.records]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, 'b-o', label='OOS Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Out-of-Sample Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.out_png, dpi=150)
            plt.close()
            print(f"[OOS] Saved plot to {self.out_png}")
        except Exception as e:
            print(f"[OOS] Could not save plot: {e}")


# ---------- Multi-Resolution Model ----------
class MultiResolutionModel(pl.LightningModule):
    """
    Shared encoder with resolution-aware losses:
      - Resolution 0 (Visium): MSE loss on soft proportions
      - Resolution 1 (Xenium): Cross-entropy loss on hard labels
    """
    
    def __init__(
        self,
        lr: float,
        n_inp: int,
        n_ctypes: int,
        n_domains: int = 0,
        token_dim: int = 256,
        adv_lambda: float = 0.0,
        recon_weight: float = 0.0,
        recon_mask_ratio: float = 0.3,
        xenium_loss_weight: float = 1.0,
        entropy_cond: bool = False,
        stage: int = 2,
        freeze_encoder_n: int = 0,
        training_mode: str = 'joint',  # 'joint', 'visium_only', 'xenium_only'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.n_inp = n_inp
        self.n_ctypes = n_ctypes
        self.n_domains = n_domains
        self.token_dim = token_dim
        self.adv_lambda_max = float(adv_lambda)
        self.recon_weight = float(recon_weight)
        self.recon_mask_ratio = float(recon_mask_ratio)
        self.xenium_loss_weight = float(xenium_loss_weight)
        self.entropy_cond = bool(entropy_cond)
        self.stage = int(stage)
        self.freeze_encoder_n = int(freeze_encoder_n)
        self.training_mode = training_mode
        
        # Shared encoder
        self.net_lat = nn.Sequential(
            FeedForward(n_inp, token_dim),
            FeedForward(token_dim, token_dim),
            FeedForward(token_dim, token_dim),
            FeedForward(token_dim, token_dim),
        )
        
        # Cell-type head
        self.ct_head_tok = nn.Linear(token_dim, n_ctypes)
        
        # Resolution embedding
        self.res_embed = nn.Embedding(num_embeddings=4, embedding_dim=token_dim)
        
        # CDAN domain head
        if n_domains > 0:
            self.grl = GradReverse(0.0)
            self.domain_head_cdan = nn.Sequential(
                nn.Linear(token_dim * n_ctypes, max(256, token_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(max(256, token_dim), n_domains),
            )
        else:
            self.grl = None
            self.domain_head_cdan = None
        
        # Reconstruction head
        self.recon_head_tok = None
        if recon_weight > 0:
            self.recon_head_tok = nn.Sequential(
                nn.Linear(token_dim, token_dim),
                nn.ReLU(inplace=True),
                nn.Linear(token_dim, n_inp),
            )
        
        self._apply_stage_freeze()
    
    def set_stage(self, stage: int, freeze_encoder_n: int = None):
        self.stage = int(stage)
        if freeze_encoder_n is not None:
            self.freeze_encoder_n = int(freeze_encoder_n)
        self._apply_stage_freeze()
    
    def set_training_mode(self, mode: str):
        """Set training mode: 'joint', 'visium_only', or 'xenium_only'"""
        assert mode in ['joint', 'visium_only', 'xenium_only']
        self.training_mode = mode
        print(f"[Model] Training mode set to: {mode}")
    
    def _apply_stage_freeze(self):
        if self.stage == 1:
            for p in self.ct_head_tok.parameters():
                p.requires_grad = False
            if self.domain_head_cdan is not None:
                for p in self.domain_head_cdan.parameters():
                    p.requires_grad = False
            for m in self.net_lat.modules():
                for p in getattr(m, 'parameters', lambda: [])():
                    p.requires_grad = True
        elif self.stage == 2:
            for p in self.ct_head_tok.parameters():
                p.requires_grad = True
            if self.domain_head_cdan is not None:
                for p in self.domain_head_cdan.parameters():
                    p.requires_grad = True
            
            ff_idx = 0
            for m in self.net_lat:
                if isinstance(m, FeedForward):
                    ff_idx += 1
                    freeze = (ff_idx <= self.freeze_encoder_n)
                    for p in m.parameters():
                        p.requires_grad = not freeze
    
    @staticmethod
    def _entropy(p):
        eps = 1e-5
        return -(p * (p + eps).log()).sum(dim=-1)
    
    def _cdan_features(self, z_tok, p_tok):
        B, T, D = z_tok.shape
        K = p_tok.shape[-1]
        zf = z_tok.view(B*T, D)
        pf = p_tok.view(B*T, K)
        outer = torch.bmm(pf.unsqueeze(2), zf.unsqueeze(1))
        outer = outer / math.sqrt(D)
        return outer.view(B, T, K*D)
    
    def inp_to_lat(self, x):
        return self.net_lat(x)
    
    def forward(self, x, resolution=None):
        z_tok = self.inp_to_lat(x)
        
        if resolution is not None:
            B, T, D = z_tok.shape
            res_emb = self.res_embed(resolution).unsqueeze(1).expand(-1, T, -1)
            z_tok = z_tok + 0.1 * res_emb
        
        logits_tok = self.ct_head_tok(z_tok)
        p_tok = torch.softmax(logits_tok, dim=-1)
        p_spot = p_tok.mean(dim=1)
        
        x_recon = None
        if self.recon_head_tok is not None:
            x_recon = self.recon_head_tok(z_tok)
        
        return p_spot, logits_tok, z_tok, x_recon
    
    def compute_loss(self, x, y, d, resolution):
        """Resolution-aware loss computation with training mode support."""
        p_spot, logits_tok, z_tok, x_recon = self.forward(x, resolution)
        
        device = x.device
        dtype = x.dtype
        
        loss_visium = torch.tensor(0.0, device=device, dtype=dtype)
        loss_xenium = torch.tensor(0.0, device=device, dtype=dtype)
        loss_domain = torch.tensor(0.0, device=device, dtype=dtype)
        loss_recon = torch.tensor(0.0, device=device, dtype=dtype)
        
        if self.stage == 2 and p_spot is not None:
            is_visium = (resolution == 0)
            is_xenium = (resolution == 1)
            
            n_visium = is_visium.sum().item()
            n_xenium = is_xenium.sum().item()
            
            # Visium MSE loss (skip if xenium_only mode)
            if n_visium > 0 and self.training_mode in ['joint', 'visium_only']:
                p_vis = p_spot[is_visium]
                y_vis = y[is_visium]
                loss_visium = F.mse_loss(p_vis, y_vis)
            
            # Xenium CE loss (skip if visium_only mode)
            if n_xenium > 0 and self.training_mode in ['joint', 'xenium_only']:
                p_xen = p_spot[is_xenium]
                y_xen = y[is_xenium]
                y_xen_class = y_xen.argmax(dim=-1)
                loss_xenium = F.nll_loss(
                    torch.log(p_xen.clamp(min=1e-8)),
                    y_xen_class
                )
        
        # CDAN domain adversarial loss
        if self.stage == 2 and self.domain_head_cdan is not None and d is not None:
            B, T, D_dim = z_tok.shape
            p_tok = torch.softmax(logits_tok, dim=-1)
            cond = self._cdan_features(z_tok, p_tok)
            logits_dom = self.domain_head_cdan(self.grl(cond))
            
            d_valid = (d >= 0)
            if d_valid.any():
                logits_flat = logits_dom[d_valid].reshape(-1, self.n_domains)
                d_rep = d[d_valid].view(-1, 1).expand(-1, T).reshape(-1)
                
                if self.entropy_cond:
                    with torch.no_grad():
                        H = self._entropy(p_tok)
                        w = 1.0 + torch.exp(-H[d_valid].reshape(-1))
                        w = w / (w.mean() + 1e-6)
                    ce = F.cross_entropy(logits_flat, d_rep.long(), reduction='none')
                    loss_domain = (ce * w).mean()
                else:
                    loss_domain = F.cross_entropy(logits_flat, d_rep.long())
        
        # Reconstruction loss
        if x_recon is not None:
            if self.recon_mask_ratio > 0:
                mask = torch.rand_like(x) < self.recon_mask_ratio
                diff = (x_recon - x)[mask]
                loss_recon = (diff ** 2).mean()
            else:
                loss_recon = F.mse_loss(x_recon, x)
        
        # Combine losses
        loss_weak = loss_visium + self.xenium_loss_weight * loss_xenium
        
        lambda_now = 0.0
        if self.stage == 2 and self.grl is not None:
            lambda_now = self.grl.lambd
        
        total_loss = loss_weak + lambda_now * loss_domain + self.recon_weight * loss_recon
        
        return {
            'loss': total_loss,
            'loss_visium': loss_visium,
            'loss_xenium': loss_xenium,
            'loss_weak': loss_weak,
            'loss_domain': loss_domain,
            'loss_recon': loss_recon,
        }
    
    def training_step(self, batch, batch_idx):
        x, y, d, r = batch
        x = x.float()
        y = y.float()
        
        losses = self.compute_loss(x, y, d, r)
        
        self.log('train_loss', losses['loss'], prog_bar=True)
        if self.stage == 2:
            self.log('train_weak_mse', losses['loss_weak'], prog_bar=True)
            self.log('train_visium_mse', losses['loss_visium'])
            self.log('train_xenium_ce', losses['loss_xenium'])
            if losses['loss_domain'] > 0:
                self.log('train_domain_cdan', losses['loss_domain'], prog_bar=True)
        if losses['loss_recon'] > 0:
            self.log('train_recon', losses['loss_recon'], prog_bar=True)
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        x, y, d, r = batch
        x = x.float()
        y = y.float()
        
        losses = self.compute_loss(x, y, d, r)
        
        self.log('val_loss', losses['loss'], prog_bar=True)
        if self.stage == 2:
            self.log('val_weak_mse', losses['loss_weak'], prog_bar=True)
            self.log('val_visium_mse', losses['loss_visium'])
            self.log('val_xenium_ce', losses['loss_xenium'])
            if losses['loss_domain'] > 0:
                self.log('val_domain_cdan', losses['loss_domain'])
        if losses['loss_recon'] > 0:
            self.log('val_recon', losses['loss_recon'])
        
        return losses['loss']
    
    def configure_optimizers(self):
        try:
            return Adam(filter(lambda p: p.requires_grad, self.parameters()),
                       lr=self.lr, fused=True)
        except TypeError:
            return Adam(filter(lambda p: p.requires_grad, self.parameters()),
                       lr=self.lr)
    
    def on_train_epoch_start(self):
        if self.stage == 2 and self.grl is not None and self.adv_lambda_max > 0:
            if self.trainer is not None:
                p = self.current_epoch / max(1, self.trainer.max_epochs - 1)
                lam = self.adv_lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
                self.grl.lambd = float(lam)
                self.log("cdan_lambda", self.grl.lambd, prog_bar=True)


# ---------- Training Functions ----------
def train_model_multi_res(
    dataset,
    batch_size: int,
    epochs: int,
    model=None,
    model_kwargs=None,
    device='cuda',
    val_dataset=None,
    save_every_n_epochs=10,
    prefix="./",
    patience=0,
    monitor_metric="val_loss",
    resume_ckpt=None,
    extra_callbacks=None,
):
    """Train multi-resolution model."""
    if model is None and model_kwargs is not None:
        model = MultiResolutionModel(**model_kwargs)
    
    num_workers = DEFAULT_NUM_WORKERS
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        )
    
    logs_dir = os.path.join(prefix, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=logs_dir, name="tb")
    csv_logger = CSVLogger(save_dir=logs_dir, name="csv")
    
    callbacks = [
        MetricTracker(),
        RichProgressBar(),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    ckpt_dir = os.path.join(prefix, "epoch_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=save_every_n_epochs,
        save_top_k=-1,
    ))
    
    if val_dataset is not None:
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=3,
        ))
    
    if patience > 0 and val_dataset is not None:
        callbacks.append(EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=patience,
        ))
    
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    
    accelerator = 'gpu' if device == 'cuda' else 'cpu'
    trainer = pl.Trainer(
        precision="bf16-mixed",
        max_epochs=epochs,
        callbacks=callbacks,
        deterministic=True,
        accelerator=accelerator,
        devices=1,
        logger=[tb_logger, csv_logger],
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        strategy="auto",
    )
    
    model.train()
    t0 = time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
    print(f"Training completed in {int(time() - t0)} seconds")
    
    tracker = callbacks[0]
    tracker.clean()
    
    return model, tracker.collection, trainer


# ---------- NEW FEATURE 1: Parallel State Training ----------
def train_single_state(state_idx: int, config: dict):
    """
    Train a single model state. Used for parallel training.
    
    Args:
        state_idx: Index of the state to train
        config: Dictionary containing all training configuration
    
    Returns:
        Path to saved checkpoint
    """
    # Unpack config
    batch_dir = config['batch_dir']
    prefix = config['prefix']
    batch_size = config['batch_size']
    epochs = config.get('epochs', 0)
    lr = config.get('lr', 1e-4)
    device = config.get('device', 'cuda')
    gpu_id = config.get('gpu_id', 0)
    val_ratio = config.get('val_ratio', 0.1)
    save_every_n_epochs = config.get('save_every_n_epochs', 10)
    n_domains = config.get('n_domains', 0)
    adv_lambda = config.get('adv_lambda', 0.0)
    recon_weight = config.get('recon_weight', 0.0)
    recon_mask_ratio = config.get('recon_mask_ratio', 0.3)
    token_dim = config.get('token_dim', 256)
    xenium_loss_weight = config.get('xenium_loss_weight', 1.0)
    monitor_metric = config.get('monitor_metric', 'val_loss')
    patience = config.get('patience', 0)
    entropy_cond = config.get('entropy_cond', False)
    two_stage = config.get('two_stage', False)
    epochs1 = config.get('epochs1', 0)
    epochs2 = config.get('epochs2', 0)
    freeze_encoder_n = config.get('freeze_encoder_n', 0)
    stage2_lr = config.get('stage2_lr', None)
    sequential_training = config.get('sequential_training', False)
    visium_epochs = config.get('visium_epochs', 0)
    xenium_epochs = config.get('xenium_epochs', 0)
    oos_sample = config.get('oos_sample', None)
    oos_tmpdir = config.get('oos_tmpdir', None)
    oos_batch_size = config.get('oos_batch_size', 64)
    
    # Set GPU
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 is our GPU
    
    print(f"\n{'='*60}")
    print(f"[State {state_idx}] Starting training on GPU {gpu_id}")
    print(f"{'='*60}")
    
    state_prefix = os.path.join(prefix, 'states', f'{state_idx:02d}/')
    os.makedirs(state_prefix, exist_ok=True)
    
    # Load dataset
    dataset = MultiResolutionDataset(batch_dir)
    
    # Train/val split
    n = len(dataset)
    np.random.seed(42 + state_idx)  # Different seed per state for variety
    indices = np.random.permutation(n)
    val_size = int(val_ratio * n)
    
    train_dataset = Subset(dataset, indices[val_size:])
    val_dataset = Subset(dataset, indices[:val_size])
    
    n_domains_eff = n_domains if n_domains > 0 else dataset.n_domains
    
    # Choose training mode
    if sequential_training:
        # Sequential: Visium first, then Xenium
        # Optionally with recon pretraining if two_stage=True
        model, _ = train_sequential(
            dataset=dataset,
            train_indices=indices[val_size:],
            val_indices=indices[:val_size],
            batch_size=batch_size,
            visium_epochs=visium_epochs,
            xenium_epochs=xenium_epochs,
            lr=lr,
            device='cuda',
            prefix=state_prefix,
            n_domains=n_domains_eff,
            adv_lambda=adv_lambda,
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=xenium_loss_weight,
            entropy_cond=entropy_cond,
            freeze_encoder_n=freeze_encoder_n,
            save_every_n_epochs=save_every_n_epochs,
            monitor_metric=monitor_metric,
            patience=patience,
            oos_sample=oos_sample,
            oos_tmpdir=oos_tmpdir,
            oos_batch_size=oos_batch_size,
            # Pass two-stage params for 3-phase training
            two_stage=two_stage,
            recon_epochs=epochs1,
            stage2_lr=stage2_lr,
        )
    elif two_stage:
        # Two-stage training
        model, _ = train_two_stage(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dataset=dataset,
            batch_size=batch_size,
            epochs1=epochs1,
            epochs2=epochs2,
            lr=lr,
            stage2_lr=stage2_lr,
            device='cuda',
            prefix=state_prefix,
            n_domains=n_domains_eff,
            adv_lambda=adv_lambda,
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=xenium_loss_weight,
            entropy_cond=entropy_cond,
            freeze_encoder_n=freeze_encoder_n,
            save_every_n_epochs=save_every_n_epochs,
            monitor_metric=monitor_metric,
            patience=patience,
            oos_sample=oos_sample,
            oos_tmpdir=oos_tmpdir,
            oos_batch_size=oos_batch_size,
        )
    else:
        # Single-stage joint training
        extra_cbs = []
        if oos_sample:
            extra_cbs.append(OOSMonitorMultiRes(
                sample_dir=oos_sample,
                prefix=state_prefix,
                tmp_dir=oos_tmpdir,
                device='cuda',
                batch_size=oos_batch_size,
            ))
        
        model_kwargs = dict(
            n_inp=dataset.n_inp,
            n_ctypes=dataset.n_out,
            n_domains=n_domains_eff,
            lr=lr,
            adv_lambda=adv_lambda,
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=xenium_loss_weight,
            entropy_cond=entropy_cond,
            stage=2,
            freeze_encoder_n=freeze_encoder_n,
        )
        
        model, _, trainer = train_model_multi_res(
            dataset=train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            model_kwargs=model_kwargs,
            device='cuda',
            val_dataset=val_dataset,
            save_every_n_epochs=save_every_n_epochs,
            prefix=state_prefix,
            patience=patience,
            monitor_metric=monitor_metric,
            extra_callbacks=extra_cbs,
        )
        
        ckpt_path = os.path.join(state_prefix, 'model.ckpt')
        trainer.save_checkpoint(ckpt_path)
    
    ckpt_path = os.path.join(state_prefix, 'model.ckpt')
    print(f"[State {state_idx}] Training complete. Saved to {ckpt_path}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return ckpt_path


def train_parallel_states(
    batch_dir: str,
    prefix: str,
    n_states: int,
    gpus: list,
    config: dict,
):
    """
    Train multiple model states in parallel across GPUs.
    
    Args:
        batch_dir: Directory containing batches
        prefix: Output directory
        n_states: Number of states to train
        gpus: List of GPU IDs to use (e.g., [0, 1, 2, 3])
        config: Training configuration dict
    
    Returns:
        List of checkpoint paths
    """
    print(f"\n{'='*60}")
    print(f"Parallel Training: {n_states} states on GPUs {gpus}")
    print(f"{'='*60}")
    
    # Create configs for each state
    state_configs = []
    for i in range(n_states):
        state_config = config.copy()
        state_config['batch_dir'] = batch_dir
        state_config['prefix'] = prefix
        state_config['gpu_id'] = gpus[i % len(gpus)]  # Round-robin GPU assignment
        state_configs.append((i, state_config))
    
    # Use multiprocessing
    mp.set_start_method('spawn', force=True)
    
    checkpoint_paths = []
    
    # Train in parallel batches (one per GPU at a time)
    batch_size = len(gpus)
    for batch_start in range(0, n_states, batch_size):
        batch_end = min(batch_start + batch_size, n_states)
        batch_configs = state_configs[batch_start:batch_end]
        
        print(f"\nTraining states {batch_start} to {batch_end - 1} in parallel...")
        
        with mp.Pool(processes=len(batch_configs)) as pool:
            results = pool.starmap(train_single_state, batch_configs)
            checkpoint_paths.extend(results)
    
    print(f"\nAll {n_states} states trained successfully!")
    return checkpoint_paths


# ---------- NEW FEATURE 2: Sequential Training (Visium -> Xenium) ----------
def train_sequential(
    dataset: MultiResolutionDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int,
    visium_epochs: int,
    xenium_epochs: int,
    lr: float,
    device: str = 'cuda',
    prefix: str = './',
    n_domains: int = 0,
    adv_lambda: float = 0.0,
    recon_weight: float = 0.0,
    recon_mask_ratio: float = 0.3,
    token_dim: int = 256,
    xenium_loss_weight: float = 1.0,
    entropy_cond: bool = False,
    freeze_encoder_n: int = 0,
    freeze_encoder_n_xenium: int = None,
    save_every_n_epochs: int = 10,
    monitor_metric: str = 'val_loss',
    patience: int = 0,
    xenium_lr: float = None,
    oos_sample: str = None,
    oos_tmpdir: str = None,
    oos_batch_size: int = 64,
    # NEW: Two-stage support (recon pretraining)
    two_stage: bool = False,
    recon_epochs: int = 0,
    stage2_lr: float = None,
):
    """
    Sequential training: Train on Visium first, then fine-tune on Xenium.
    
    Optionally supports 3-phase training when two_stage=True:
      Phase 0: Reconstruction pretraining (recon_epochs)
      Phase 1: Visium training (visium_epochs)
      Phase 2: Xenium fine-tuning (xenium_epochs)
    
    Args:
        dataset: Full MultiResolutionDataset
        train_indices: Indices for training set
        val_indices: Indices for validation set
        visium_epochs: Number of epochs for Visium training
        xenium_epochs: Number of epochs for Xenium fine-tuning
        freeze_encoder_n_xenium: Layers to freeze during Xenium phase (default: same as freeze_encoder_n)
        xenium_lr: Learning rate for Xenium phase (default: lr/10)
        two_stage: If True, add reconstruction pretraining before Visium
        recon_epochs: Number of epochs for reconstruction pretraining (Stage 1)
        stage2_lr: Learning rate for Stage 2 (Visium + Xenium phases)
    """
    phase_info = "Visium -> Xenium"
    if two_stage and recon_epochs > 0:
        phase_info = f"Recon ({recon_epochs}ep) -> Visium ({visium_epochs}ep) -> Xenium ({xenium_epochs}ep)"
    else:
        phase_info = f"Visium ({visium_epochs}ep) -> Xenium ({xenium_epochs}ep)"
    
    print(f"\n{'='*60}")
    print(f"Sequential Training: {phase_info}")
    print(f"{'='*60}")
    
    if freeze_encoder_n_xenium is None:
        freeze_encoder_n_xenium = freeze_encoder_n
    if xenium_lr is None:
        xenium_lr = lr / 10  # Lower LR for fine-tuning
    
    # Create filtered datasets
    visium_train = FilteredResolutionDataset(
        Subset(dataset, train_indices).dataset if isinstance(Subset(dataset, train_indices), Subset) else dataset,
        resolution=0
    )
    visium_val = FilteredResolutionDataset(
        Subset(dataset, val_indices).dataset if isinstance(Subset(dataset, val_indices), Subset) else dataset,
        resolution=0
    )
    
    # For Xenium, we need to create proper subsets
    # First get the base dataset
    base_dataset = dataset
    
    # Create train/val subsets that filter by resolution
    train_visium_indices = [
        i for i in train_indices
        if base_dataset.samples[i][2] == 0  # resolution == 0
    ]
    val_visium_indices = [
        i for i in val_indices
        if base_dataset.samples[i][2] == 0
    ]
    train_xenium_indices = [
        i for i in train_indices
        if base_dataset.samples[i][2] == 1  # resolution == 1
    ]
    val_xenium_indices = [
        i for i in val_indices
        if base_dataset.samples[i][2] == 1
    ]
    
    visium_train_dataset = Subset(base_dataset, train_visium_indices)
    visium_val_dataset = Subset(base_dataset, val_visium_indices) if val_visium_indices else None
    xenium_train_dataset = Subset(base_dataset, train_xenium_indices)
    xenium_val_dataset = Subset(base_dataset, val_xenium_indices) if val_xenium_indices else None
    
    print(f"  Visium train: {len(visium_train_dataset)}, val: {len(visium_val_dataset) if visium_val_dataset else 0}")
    print(f"  Xenium train: {len(xenium_train_dataset)}, val: {len(xenium_val_dataset) if xenium_val_dataset else 0}")
    
    # Initialize model
    model = None
    
    # ========== Phase 0: Reconstruction Pretraining (if two_stage) ==========
    if two_stage and recon_epochs > 0:
        recon_prefix = os.path.join(prefix, 'recon_phase/')
        os.makedirs(recon_prefix, exist_ok=True)
        
        print(f"\n--- Phase 0: Reconstruction Pretraining ({recon_epochs} epochs) ---")
        
        # Use full dataset for reconstruction (both Visium and Xenium)
        full_train_dataset = Subset(base_dataset, train_indices)
        full_val_dataset = Subset(base_dataset, val_indices) if len(val_indices) > 0 else None
        
        model = MultiResolutionModel(
            n_inp=dataset.n_inp,
            n_ctypes=dataset.n_out,
            n_domains=n_domains,
            lr=lr,
            adv_lambda=0.0,  # No adversary in recon phase
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=0.0,
            entropy_cond=False,
            stage=1,  # Stage 1 = recon only
            freeze_encoder_n=0,
            training_mode='joint',
        )
        
        extra_cbs_recon = []
        if oos_sample:
            extra_cbs_recon.append(OOSMonitorMultiRes(
                sample_dir=oos_sample,
                prefix=recon_prefix,
                tmp_dir=oos_tmpdir,
                device=device,
                batch_size=oos_batch_size,
                resolution=0,
            ))
        
        model, _, trainer0 = train_model_multi_res(
            dataset=full_train_dataset,
            batch_size=batch_size,
            epochs=recon_epochs,
            model=model,
            device=device,
            val_dataset=full_val_dataset,
            save_every_n_epochs=save_every_n_epochs,
            prefix=recon_prefix,
            patience=patience,
            monitor_metric='val_loss',
            extra_callbacks=extra_cbs_recon,
        )
        
        recon_ckpt = os.path.join(recon_prefix, 'model_recon.ckpt')
        trainer0.save_checkpoint(recon_ckpt)
        print(f"Reconstruction phase complete. Saved to {recon_ckpt}")
        
        # Transition to Stage 2
        model.set_stage(2, freeze_encoder_n=freeze_encoder_n)
        if stage2_lr:
            model.lr = stage2_lr
    
    # ========== Phase 1: Visium Training ==========
    visium_prefix = os.path.join(prefix, 'visium_phase/')
    os.makedirs(visium_prefix, exist_ok=True)
    
    print(f"\n--- Phase 1: Visium Training ({visium_epochs} epochs) ---")
    
    # Create model if not already created from recon phase
    if model is None:
        model_kwargs = dict(
            n_inp=dataset.n_inp,
            n_ctypes=dataset.n_out,
            n_domains=n_domains,
            lr=lr,
            adv_lambda=adv_lambda,
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=0.0,  # Disable Xenium loss in phase 1
            entropy_cond=entropy_cond,
            stage=2,
            freeze_encoder_n=freeze_encoder_n,
            training_mode='visium_only',
        )
        model = MultiResolutionModel(**model_kwargs)
    else:
        # Model from recon phase - update for Visium training
        model.xenium_loss_weight = 0.0
        model.adv_lambda_max = float(adv_lambda)
        model.entropy_cond = bool(entropy_cond)
        model.set_training_mode('visium_only')
    
    extra_cbs_vis = []
    if oos_sample:
        extra_cbs_vis.append(OOSMonitorMultiRes(
            sample_dir=oos_sample,
            prefix=visium_prefix,
            tmp_dir=oos_tmpdir,
            device=device,
            batch_size=oos_batch_size,
            resolution=0,
        ))
    
    model, _, trainer1 = train_model_multi_res(
        dataset=visium_train_dataset,
        batch_size=batch_size,
        epochs=visium_epochs,
        model=model,
        device=device,
        val_dataset=visium_val_dataset,
        save_every_n_epochs=save_every_n_epochs,
        prefix=visium_prefix,
        patience=patience,
        monitor_metric='val_visium_mse',
        extra_callbacks=extra_cbs_vis,
    )
    
    visium_ckpt = os.path.join(visium_prefix, 'model_visium.ckpt')
    trainer1.save_checkpoint(visium_ckpt)
    print(f"Visium phase complete. Saved to {visium_ckpt}")
    
    # ========== Phase 2: Xenium Fine-tuning ==========
    if xenium_epochs > 0 and len(xenium_train_dataset) > 0:
        xenium_prefix = os.path.join(prefix, 'xenium_phase/')
        os.makedirs(xenium_prefix, exist_ok=True)
        
        print(f"\n--- Phase 2: Xenium Fine-tuning ({xenium_epochs} epochs) ---")
        
        # Update model for Xenium phase
        model.set_training_mode('xenium_only')
        model.lr = xenium_lr
        model.xenium_loss_weight = xenium_loss_weight
        model.freeze_encoder_n = freeze_encoder_n_xenium
        model._apply_stage_freeze()
        
        extra_cbs_xen = []
        if oos_sample:
            extra_cbs_xen.append(OOSMonitorMultiRes(
                sample_dir=oos_sample,
                prefix=xenium_prefix,
                tmp_dir=oos_tmpdir,
                device=device,
                batch_size=oos_batch_size,
                resolution=1,
            ))
        
        model, _, trainer2 = train_model_multi_res(
            dataset=xenium_train_dataset,
            batch_size=batch_size,
            epochs=xenium_epochs,
            model=model,
            device=device,
            val_dataset=xenium_val_dataset,
            save_every_n_epochs=save_every_n_epochs,
            prefix=xenium_prefix,
            patience=patience,
            monitor_metric='val_xenium_ce',
            extra_callbacks=extra_cbs_xen,
        )
        
        xenium_ckpt = os.path.join(xenium_prefix, 'model_xenium.ckpt')
        trainer2.save_checkpoint(xenium_ckpt)
        print(f"Xenium phase complete. Saved to {xenium_ckpt}")
    
    # Save final model
    final_ckpt = os.path.join(prefix, 'model.ckpt')
    trainer = trainer2 if xenium_epochs > 0 and len(xenium_train_dataset) > 0 else trainer1
    trainer.save_checkpoint(final_ckpt)
    print(f"Final model saved to {final_ckpt}")
    
    # Reset training mode for prediction
    model.set_training_mode('joint')
    model.eval()
    
    return model, dataset


def train_two_stage(
    train_dataset,
    val_dataset,
    dataset,
    batch_size: int,
    epochs1: int,
    epochs2: int,
    lr: float,
    stage2_lr: float = None,
    device: str = 'cuda',
    prefix: str = './',
    n_domains: int = 0,
    adv_lambda: float = 0.0,
    recon_weight: float = 0.0,
    recon_mask_ratio: float = 0.3,
    token_dim: int = 256,
    xenium_loss_weight: float = 1.0,
    entropy_cond: bool = False,
    freeze_encoder_n: int = 0,
    save_every_n_epochs: int = 10,
    monitor_metric: str = 'val_loss',
    patience: int = 0,
    oos_sample: str = None,
    oos_tmpdir: str = None,
    oos_batch_size: int = 64,
):
    """Two-stage training: Stage 1 (recon only) -> Stage 2 (weak + CDAN)."""
    
    # Stage 1: Reconstruction only
    stage1_prefix = os.path.join(prefix, 'stage1/')
    os.makedirs(stage1_prefix, exist_ok=True)
    
    model_stage1 = MultiResolutionModel(
        n_inp=dataset.n_inp,
        n_ctypes=dataset.n_out,
        n_domains=n_domains,
        lr=lr,
        adv_lambda=0.0,
        recon_weight=recon_weight,
        recon_mask_ratio=recon_mask_ratio,
        token_dim=token_dim,
        xenium_loss_weight=xenium_loss_weight,
        entropy_cond=False,
        stage=1,
        freeze_encoder_n=0,
    )
    
    extra_cbs1 = []
    if oos_sample:
        extra_cbs1.append(OOSMonitorMultiRes(
            sample_dir=oos_sample,
            prefix=stage1_prefix,
            tmp_dir=oos_tmpdir,
            device=device,
            batch_size=oos_batch_size,
        ))
    
    model_stage1, hist1, trainer1 = train_model_multi_res(
        dataset=train_dataset,
        batch_size=batch_size,
        epochs=epochs1,
        model=model_stage1,
        device=device,
        val_dataset=val_dataset,
        save_every_n_epochs=save_every_n_epochs,
        prefix=stage1_prefix,
        patience=patience,
        monitor_metric='val_loss',
        extra_callbacks=extra_cbs1,
    )
    
    ckpt1 = os.path.join(stage1_prefix, 'model_stage1.ckpt')
    trainer1.save_checkpoint(ckpt1)
    print(f'[two-stage] Stage 1 saved to {ckpt1}')
    
    # Stage 2: Weak + CDAN
    stage2_prefix = os.path.join(prefix, 'stage2/')
    os.makedirs(stage2_prefix, exist_ok=True)
    
    model_stage1.set_stage(2, freeze_encoder_n=freeze_encoder_n)
    model_stage2 = model_stage1
    model_stage2.lr = stage2_lr if stage2_lr else lr
    model_stage2.adv_lambda_max = float(adv_lambda)
    model_stage2.entropy_cond = bool(entropy_cond)
    
    extra_cbs2 = []
    if oos_sample:
        extra_cbs2.append(OOSMonitorMultiRes(
            sample_dir=oos_sample,
            prefix=stage2_prefix,
            tmp_dir=oos_tmpdir,
            device=device,
            batch_size=oos_batch_size,
        ))
    
    model_stage2, hist2, trainer2 = train_model_multi_res(
        dataset=train_dataset,
        batch_size=batch_size,
        epochs=epochs2,
        model=model_stage2,
        device=device,
        val_dataset=val_dataset,
        save_every_n_epochs=save_every_n_epochs,
        prefix=stage2_prefix,
        patience=patience,
        monitor_metric=monitor_metric,
        extra_callbacks=extra_cbs2,
    )
    
    final_ckpt = os.path.join(prefix, 'model.ckpt')
    trainer2.save_checkpoint(final_ckpt)
    print(f'[two-stage] Final model saved to {final_ckpt}')
    
    model_stage2.eval()
    return model_stage2, dataset


# ---------- Main Training Function ----------
def get_model_batched_multi_res(
    batch_dir: str,
    prefix: str,
    batch_size: int,
    epochs: int,
    lr: float,
    load_saved=False,
    device='cuda',
    val_ratio=0.1,
    save_every_n_epochs=10,
    n_domains: int = 0,
    adv_lambda: float = 0.0,
    recon_weight: float = 0.0,
    recon_mask_ratio: float = 0.3,
    token_dim: int = 256,
    xenium_loss_weight: float = 1.0,
    monitor_metric='val_loss',
    patience=0,
    resume_ckpt=None,
    entropy_cond: bool = False,
    two_stage: bool = False,
    epochs1: int = 0,
    epochs2: int = 0,
    freeze_encoder_n: int = 0,
    stage2_lr: float = None,
    resolution_map: dict = None,
    sequential_training: bool = False,
    visium_epochs: int = 0,
    xenium_epochs: int = 0,
    xenium_lr: float = None,
    freeze_encoder_n_xenium: int = None,
    oos_sample: str = None,
    oos_tmpdir: str = None,
    oos_batch_size: int = 64,
):
    """Main training function with all modes supported."""
    print(f'Loading multi-resolution dataset from: {batch_dir}')
    dataset = MultiResolutionDataset(batch_dir, resolution_map=resolution_map)
    print(f'Dataset: {len(dataset)} samples, T={dataset.token_len}, C={dataset.n_inp}, K={dataset.n_out}')
    
    # Train/val split
    n = len(dataset)
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_size = int(val_ratio * n)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    
    n_domains_eff = n_domains if n_domains > 0 else dataset.n_domains
    print(f'Using n_domains={n_domains_eff}')
    
    # Choose training mode
    if sequential_training:
        # Sequential: Visium first, then Xenium
        # Optionally with recon pretraining if two_stage=True
        model, dataset = train_sequential(
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            batch_size=batch_size,
            visium_epochs=visium_epochs,
            xenium_epochs=xenium_epochs,
            lr=lr,
            device=device,
            prefix=prefix,
            n_domains=n_domains_eff,
            adv_lambda=adv_lambda,
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=xenium_loss_weight,
            entropy_cond=entropy_cond,
            freeze_encoder_n=freeze_encoder_n,
            freeze_encoder_n_xenium=freeze_encoder_n_xenium,
            save_every_n_epochs=save_every_n_epochs,
            monitor_metric=monitor_metric,
            patience=patience,
            xenium_lr=xenium_lr,
            oos_sample=oos_sample,
            oos_tmpdir=oos_tmpdir,
            oos_batch_size=oos_batch_size,
            # Pass two-stage params for 3-phase training
            two_stage=two_stage,
            recon_epochs=epochs1,
            stage2_lr=stage2_lr,
        )
        return model, dataset
    
    elif two_stage:
        # Two-stage training
        model, dataset = train_two_stage(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dataset=dataset,
            batch_size=batch_size,
            epochs1=epochs1,
            epochs2=epochs2,
            lr=lr,
            stage2_lr=stage2_lr,
            device=device,
            prefix=prefix,
            n_domains=n_domains_eff,
            adv_lambda=adv_lambda,
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=xenium_loss_weight,
            entropy_cond=entropy_cond,
            freeze_encoder_n=freeze_encoder_n,
            save_every_n_epochs=save_every_n_epochs,
            monitor_metric=monitor_metric,
            patience=patience,
            oos_sample=oos_sample,
            oos_tmpdir=oos_tmpdir,
            oos_batch_size=oos_batch_size,
        )
        return model, dataset
    
    else:
        # Single-stage joint training
        extra_cbs = []
        if oos_sample:
            extra_cbs.append(OOSMonitorMultiRes(
                sample_dir=oos_sample,
                prefix=prefix,
                tmp_dir=oos_tmpdir,
                device=device,
                batch_size=oos_batch_size,
            ))
        
        model_kwargs = dict(
            n_inp=dataset.n_inp,
            n_ctypes=dataset.n_out,
            n_domains=n_domains_eff,
            lr=lr,
            adv_lambda=adv_lambda,
            recon_weight=recon_weight,
            recon_mask_ratio=recon_mask_ratio,
            token_dim=token_dim,
            xenium_loss_weight=xenium_loss_weight,
            entropy_cond=entropy_cond,
            stage=2,
            freeze_encoder_n=freeze_encoder_n,
        )
        
        model, history, trainer = train_model_multi_res(
            dataset=train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            model_kwargs=model_kwargs,
            device=device,
            val_dataset=val_dataset,
            save_every_n_epochs=save_every_n_epochs,
            prefix=prefix,
            patience=patience,
            monitor_metric=monitor_metric,
            resume_ckpt=resume_ckpt,
            extra_callbacks=extra_cbs,
        )
        
        ckpt_path = os.path.join(prefix, 'model.ckpt')
        trainer.save_checkpoint(ckpt_path)
        print(f'Model saved to {ckpt_path}')
        
        model.eval()
        if device == 'cuda':
            torch.cuda.empty_cache()
        return model, dataset


# ---------- Prediction ----------
def load_model_for_prediction(checkpoint_path: str, device='cuda'):
    """Load a MultiResolutionModel checkpoint for prediction."""
    model = MultiResolutionModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def predict_batched_multi_res(model_states, batch_dir, prefix, device='cuda', resolution_map=None):
    """Export token-level latents (median across states)."""
    model_states = [m.to(device) for m in model_states]
    
    dataset = MultiResolutionDataset(batch_dir, resolution_map=resolution_map)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, 
                       num_workers=0, collate_fn=collate_fn)
    
    z_all = []
    print("Predicting latent features...")
    
    model0 = model_states[0]
    dtype = next(model0.parameters()).dtype
    
    with torch.no_grad():
        for batch_idx, (x, y, d, r) in enumerate(loader):
            print(f"  Batch {batch_idx + 1}/{len(loader)}")
            x = x.to(device=device, dtype=dtype)
            r = r.to(device=device)
            
            z_batch_states = []
            for model in model_states:
                z = model.inp_to_lat(x)
                z_batch_states.append(z.cpu())
            
            z_median = torch.median(torch.stack(z_batch_states, dim=0), dim=0).values
            z_all.append(z_median.numpy())
    
    z_point = np.concatenate(z_all, axis=0)
    z_dict = dict(cls=z_point.transpose(2, 0, 1))
    save_pickle(z_dict, os.path.join(prefix, 'embeddings-gene.pickle'))
    print(f"Saved: {z_point.shape} -> {prefix}embeddings-gene.pickle")


# ---------- CLI ----------
def get_args():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-resolution training (Visium + Xenium)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard joint training
    python train_multi_resolution_enhanced.py /path/to/batches --epochs 200 --n-states 2

    # Two-stage training (recon pretraining + joint Visium/Xenium)
    python train_multi_resolution_enhanced.py /path/to/batches \\
        --two-stage --epochs1 15 --epochs2 200 --n-states 2

    # Sequential training (Visium first, then Xenium)
    python train_multi_resolution_enhanced.py /path/to/batches \\
        --sequential-training --visium-epochs 100 --xenium-epochs 50 --n-states 2

    # 3-PHASE TRAINING: Recon -> Visium -> Xenium (combine --two-stage + --sequential-training)
    python train_multi_resolution_enhanced.py /path/to/batches \\
        --two-stage --epochs1 15 \\
        --sequential-training --visium-epochs 100 --xenium-epochs 50 \\
        --n-states 2 --freeze-encoder-n 2 --recon-weight 0.1

    # Parallel state training on multiple GPUs
    python train_multi_resolution_enhanced.py /path/to/batches \\
        --two-stage --epochs1 15 --epochs2 200 \\
        --n-states 4 --parallel-states --gpus 0,1,2,3
        """
    )
    parser.add_argument('prefix', type=str, help='Working directory with batches/ subfolder')
    
    # Basic training
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--n-states', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load-saved', action='store_true')
    parser.add_argument('--save-every-n-epochs', type=int, default=10)
    
    # Model architecture
    parser.add_argument('--token-dim', type=int, default=256)
    parser.add_argument('--xenium-weight', type=float, default=0.01,
                       help='Weight for Xenium CE loss relative to Visium MSE')
    
    # CDAN / Domain
    parser.add_argument('--n-domains', type=int, default=0)
    parser.add_argument('--adv-lambda', type=float, default=0.0)
    parser.add_argument('--entropy-cond', action='store_true')
    
    # Reconstruction
    parser.add_argument('--recon-weight', type=float, default=0.1)
    parser.add_argument('--recon-mask-ratio', type=float, default=0.3)
    
    # Two-stage training
    parser.add_argument('--two-stage', action='store_true',
                       help='Add reconstruction pretraining stage. Can combine with --sequential-training for 3-phase training.')
    parser.add_argument('--epochs1', type=int, default=15,
                       help='Epochs for recon pretraining (Stage 1). Used with --two-stage.')
    parser.add_argument('--epochs2', type=int, default=0,
                       help='Epochs for Stage 2 (only used if NOT using --sequential-training)')
    parser.add_argument('--freeze-encoder-n', type=int, default=2)
    parser.add_argument('--stage2-lr', type=float, default=None)
    
    # NEW: Sequential training (Visium -> Xenium)
    parser.add_argument('--sequential-training', action='store_true',
                       help='Train on Visium first, then fine-tune on Xenium. '
                            'Can combine with --two-stage for 3-phase: Recon -> Visium -> Xenium')
    parser.add_argument('--visium-epochs', type=int, default=100,
                       help='Number of epochs for Visium phase (sequential mode)')
    parser.add_argument('--xenium-epochs', type=int, default=100,
                       help='Number of epochs for Xenium phase (sequential mode)')
    parser.add_argument('--xenium-lr', type=float, default=None,
                       help='Learning rate for Xenium phase (default: lr/10)')
    parser.add_argument('--freeze-encoder-n-xenium', type=int, default=None,
                       help='Encoder layers to freeze during Xenium phase')
    
    # NEW: Parallel state training
    parser.add_argument('--parallel-states', action='store_true',
                       help='Train multiple states in parallel across GPUs')
    parser.add_argument('--gpus', type=str, default='0',
                       help='Comma-separated list of GPU IDs (e.g., "0,1,2,3")')
    
    # Checkpointing
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--monitor-metric', type=str, default='val_weak_mse')
    parser.add_argument('--resume-ckpt', type=str, default=None)
    
    # OOS monitoring
    parser.add_argument('--oos-sample', type=str, default=None)
    parser.add_argument('--oos-tmpdir', type=str, default=None)
    parser.add_argument('--oos-batch-size', type=int, default=64)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Parse GPU list
    gpus = [int(g.strip()) for g in args.gpus.split(',')]
    
    # Determine batch directory
    batch_dir = os.path.join(args.prefix, 'batches')
    if not os.path.exists(batch_dir):
        if any(f.startswith('batch_') and f.endswith('_x.npy') 
               for f in os.listdir(args.prefix)):
            batch_dir = args.prefix
        else:
            print(f"ERROR: No batches found in {args.prefix} or {batch_dir}")
            return
    
    print(f"Using batch directory: {batch_dir}")
    
    # Auto batch size
    sample_dataset = MultiResolutionDataset(batch_dir)
    batch_size = auto_batch_size(sample_dataset, target_tokens=300000, hard_cap=128)
    del sample_dataset
    gc.collect()
    
    print(f"Using batch_size={batch_size}")
    
    # Build config for parallel training
    config = {
        'batch_dir': batch_dir,
        'prefix': args.prefix,
        'batch_size': batch_size,
        'epochs': args.epochs or 0,
        'lr': 1e-4,
        'device': args.device,
        'val_ratio': 0.1,
        'save_every_n_epochs': args.save_every_n_epochs,
        'n_domains': args.n_domains,
        'adv_lambda': args.adv_lambda,
        'recon_weight': args.recon_weight,
        'recon_mask_ratio': args.recon_mask_ratio,
        'token_dim': args.token_dim,
        'xenium_loss_weight': args.xenium_weight,
        'monitor_metric': args.monitor_metric,
        'patience': args.patience,
        'entropy_cond': args.entropy_cond,
        'two_stage': args.two_stage,
        'epochs1': args.epochs1,
        'epochs2': args.epochs2,
        'freeze_encoder_n': args.freeze_encoder_n,
        'stage2_lr': args.stage2_lr,
        'sequential_training': args.sequential_training,
        'visium_epochs': args.visium_epochs,
        'xenium_epochs': args.xenium_epochs,
        'xenium_lr': args.xenium_lr,
        'freeze_encoder_n_xenium': args.freeze_encoder_n_xenium,
        'oos_sample': args.oos_sample,
        'oos_tmpdir': args.oos_tmpdir,
        'oos_batch_size': args.oos_batch_size,
    }
    
    # Choose training mode
    if args.parallel_states and args.n_states > 1:
        # Parallel training across GPUs
        checkpoint_paths = train_parallel_states(
            batch_dir=batch_dir,
            prefix=args.prefix,
            n_states=args.n_states,
            gpus=gpus,
            config=config,
        )
        
        # Load models for prediction
        model_states = [
            load_model_for_prediction(ckpt, device=args.device)
            for ckpt in checkpoint_paths
        ]
    else:
        # Sequential state training (original behavior)
        model_states = []
        for i in range(args.n_states):
            print(f"\n{'='*60}")
            print(f"Training model state {i+1}/{args.n_states}")
            print(f"{'='*60}")
            
            state_prefix = os.path.join(args.prefix, 'states', f'{i:02d}/')
            
            model, dataset = get_model_batched_multi_res(
                batch_dir=batch_dir,
                prefix=state_prefix,
                batch_size=batch_size,
                epochs=(0 if args.two_stage or args.sequential_training else (args.epochs or 0)),
                lr=1e-4,
                load_saved=args.load_saved,
                device=args.device,
                val_ratio=0.1,
                save_every_n_epochs=args.save_every_n_epochs,
                n_domains=args.n_domains,
                adv_lambda=args.adv_lambda,
                recon_weight=args.recon_weight,
                recon_mask_ratio=args.recon_mask_ratio,
                token_dim=args.token_dim,
                xenium_loss_weight=args.xenium_weight,
                monitor_metric=args.monitor_metric,
                patience=args.patience,
                resume_ckpt=args.resume_ckpt,
                entropy_cond=args.entropy_cond,
                two_stage=args.two_stage,
                epochs1=args.epochs1,
                epochs2=args.epochs2,
                freeze_encoder_n=args.freeze_encoder_n,
                stage2_lr=args.stage2_lr,
                sequential_training=args.sequential_training,
                visium_epochs=args.visium_epochs,
                xenium_epochs=args.xenium_epochs,
                xenium_lr=args.xenium_lr,
                freeze_encoder_n_xenium=args.freeze_encoder_n_xenium,
                oos_sample=args.oos_sample,
                oos_tmpdir=args.oos_tmpdir,
                oos_batch_size=args.oos_batch_size,
            )
            
            model_states.append(model)
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    
    # Prediction
    print("\n" + "="*60)
    print("Starting prediction...")
    print("="*60)
    predict_batched_multi_res(
        model_states=model_states,
        batch_dir=batch_dir,
        prefix=args.prefix,
        device=args.device,
    )
    
    print("\nTraining and prediction completed!")


if __name__ == '__main__':
    main()