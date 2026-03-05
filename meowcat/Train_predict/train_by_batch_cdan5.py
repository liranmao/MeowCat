# update for more data training for cdan with no l2 and model structure change
# add the oos loss calculation and visualization
# change the cdan loss calculation
# ============================================
# Weakly-supervised (per-token) + CDAN (per-token) + Recon (per-token)
# Two-stage training support
#   Stage 1: no cell-type head / weak loss / adversary (self-supervised pretrain via recon)
#   Stage 2: enable cell-type head + weak supervision + CDAN adversary, with optional partial freeze
#
# Inputs (same as before):
#   x: [B, T, C] token features per spot (T = pixels-in-disk, C = feature dim)
#   y_soft: [B, K] spot-level cell-type proportions
#   d_lbl: [B] domain id (WSI id)
#
# Losses (Stage 2):
#   L_weak  = MSE( mean_token(softmax(ct_logits_tok)), y_soft )
#   L_CDAN  = CE( D(GRL( z_tok ⊗ p_tok )), d_lbl )   averaged over tokens
#               (optional entropy conditioning on per-token softmax)
#   L_recon = MSE on masked tokens (optional; used in both stages if enabled)
#
# CLI examples:
#   # single-stage (backward compatible; CDAN disabled by default)
#   python train_cdan.py PREFIX --epochs 30 --adv-lambda 0.1 --recon-weight 0.1 --recon-mask-ratio 0.3
#
#   # two-stage: pretrain (recon only) then enable weak+CDAN with partial freeze
#   python train_cdan.py PREFIX --two-stage --epochs1 10 --epochs2 20 \
#       --adv-lambda 0.1 --entropy-cond \
#       --freeze-encoder-n 2 --recon-weight 0.1 --recon-mask-ratio 0.3
# ============================================
import pandas as pd
import argparse
import os
import sys
import torch

# Ensure sibling directories are importable (Preprocess/ and Train_predict/ itself)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_preprocess_dir = os.path.join(os.path.dirname(_this_dir), "Preprocess")
for _d in (_this_dir, _preprocess_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import numpy as np
import gc
import pickle
from time import time
from torch.autograd import Function
import shutil
from impute_by_basic import get_gene_counts, get_locs
from utils import read_lines, read_string, save_pickle, load_pickle
from image import get_disk_mask
import math
from bisect import bisect_right
import json

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.progress import RichProgressBar
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_FACTOR = 4

import warnings
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors",
    category=UserWarning,
    module="torch.utils.data._utils.collate"
)

# ---------- Utilities ----------
# --- NEW: OOS monitor callback (per-epoch) ---
class OOSMonitor(pl.Callback):
    """
    Evaluates a single out-of-sample (OOS) sample at the end of each validation epoch,
    logs metrics to Lightning loggers, and saves a PNG curve + TSV at the end of fit.
    Requires the helper functions: _ensure_oos_batch(), evaluate_oos_sample()
    already added earlier.
    """
    def __init__(self, sample_dir: str, prefix: str,
                 tmp_dir: str = None, radius: int = None,
                 device: str = 'cuda', batch_size: int = 0):
        super().__init__()
        self.sample_dir = sample_dir
        self.tmp_dir = tmp_dir
        self.radius = radius
        self.device = device
        self.batch_size = int(batch_size) if batch_size else 0
        self.prefix = prefix
        self.records = []  # list of dicts per epoch

        # output files
        self.out_png = os.path.join(self.prefix, "oos_loss_curve.png")
        self.out_tsv = os.path.join(self.prefix, "oos_loss_curve.tsv")

    
    def setup(self, trainer, pl_module, stage: str):
        self.oos_batch_dir = _ensure_oos_batch(self.sample_dir, self.tmp_dir, self.radius)

        ds = CachedBatchDataset(self.oos_batch_dir, max_cached_batches=2)
        if self.batch_size <= 0:
            self.batch_size = auto_batch_size(ds, target_tokens=300000, hard_cap=128)

        # NEW: keep dataset/loader; avoid persistent workers for a short eval job
        self._oos_dataset = ds
        self._oos_loader = DataLoader(
            self._oos_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,              # <— important: no worker pool => no extra FDs
            pin_memory=False,           # <— avoid pin thread for this small eval
            persistent_workers=False    # <— critical: don’t leak workers each epoch
            # prefetch_factor not used when num_workers=0
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        dev = 'cuda' if (hasattr(pl_module, 'device') and getattr(pl_module.device, 'type', 'cpu') == 'cuda') else 'cpu'
        dev = self.device or dev

        res = evaluate_oos_sample(
            model=pl_module,
            oos_batch_dir=self.oos_batch_dir,
            device=dev,
            batch_size=self.batch_size,
            # NEW: pass the prebuilt loader to avoid recreating it
            loader=self._oos_loader
        )
        # Log to Lightning (shows up in TensorBoard & CSV)
        pl_module.log('oos_val_loss',    res['val_loss_oos'],   prog_bar=True, on_step=False, on_epoch=True)
        pl_module.log('oos_weak_mse',    res['weak_mse_oos'],   on_step=False, on_epoch=True)
        pl_module.log('oos_domain_cdan', res['domain_cdan_oos'],on_step=False, on_epoch=True)
        pl_module.log('oos_recon_tok',   res['recon_tok_oos'],  on_step=False, on_epoch=True)

        self.records.append(dict(epoch=int(trainer.current_epoch), **res))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Persist TSV
        os.makedirs(self.prefix, exist_ok=True)
        with open(self.out_tsv, "w") as f:
            f.write("epoch\tval_loss_oos\tweak_mse_oos\tdomain_cdan_oos\trecon_tok_oos\tn_samples\n")
            for r in self.records:
                f.write(
                    f"{r['epoch']}\t{r['val_loss_oos']}\t{r['weak_mse_oos']}\t"
                    f"{r['domain_cdan_oos']}\t{r['recon_tok_oos']}\t{r['n_samples']}\n"
                )

        # Plot PNG (best-effort)
        try:
            import matplotlib.pyplot as plt
            epochs = [r['epoch'] for r in self.records]
            losses = [r['val_loss_oos'] for r in self.records]

            plt.figure()
            plt.plot(epochs, losses, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("OOS validation loss")
            plt.title("Out-of-sample (OOS) loss vs. epochs")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.out_png, dpi=150)
            plt.close()
            print(f"[OOS] Saved curve → {self.out_png}")
        except Exception as e:
            print(f"[OOS] Could not render PNG plot ({e}). TSV saved at {self.out_tsv}.")

# --- OOS helpers (needed by OOSMonitor) ---
def _ensure_oos_batch(sample_dir: str, tmp_dir: str = None, radius: int = None):
    """
    Build (or reuse) a single-sample batch dir from sample_dir.
    """
    if tmp_dir is None:
        tmp_dir = os.path.join(sample_dir, "_oos_batch")
    os.makedirs(tmp_dir, exist_ok=True)

    have_batches = any(
        (f.startswith("batch_") and (f.endswith("_x.npy") or f.endswith(".pkl")))
        for f in os.listdir(tmp_dir)
    )
    if not have_batches:
        r = radius if radius is not None else _read_radius(sample_dir)
        if r is None:
            raise RuntimeError(f"Could not read radius from {sample_dir}/radius.txt")
        prepare_and_save_batches([sample_dir], r, tmp_dir, samples_per_batch=1, domain_map_file=None)
    return tmp_dir


@torch.no_grad()
def evaluate_oos_sample(model, oos_batch_dir, device='cuda', batch_size=None, loader=None):
    model.eval().to(device)
    if loader is None:
        dataset = CachedBatchDataset(oos_batch_dir, max_cached_batches=2)
        if batch_size is None or batch_size <= 0:
            batch_size = auto_batch_size(dataset, target_tokens=300000, hard_cap=128)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0,                 # <— keep 0 here too if you ever fall back
            pin_memory=False,
            persistent_workers=False
        )
    
    dtype = next(model.parameters()).dtype

    total_N = 0
    sum_loss = sum_weak = sum_dom = sum_recon = 0.0

    for xb, yb, db in loader:
        B = xb.shape[0]; total_N += B
        xb = xb.to(device=device, dtype=dtype)
        yb = yb.to(device=device, dtype=dtype)
        db = db.to(device=device, dtype=torch.long)

        p_spot, logits_dom_tok, x_recon_tok, z_tok = model.forward(xb)

        # weak
        loss_weak = torch.tensor(0.0, device=device, dtype=dtype)
        if model.stage == 2 and p_spot is not None:
            loss_weak = F.mse_loss(p_spot, yb)

        # domain
        loss_domain = torch.tensor(0.0, device=device, dtype=dtype)
        if model.stage == 2 and logits_dom_tok is not None:
            Bt, Tt, Dt = logits_dom_tok.shape
            d_valid = (db >= 0)
            if torch.any(d_valid):
                logits_flat = logits_dom_tok[d_valid].reshape(-1, Dt)
                d_rep = db[d_valid].view(-1, 1).expand(-1, Tt).reshape(-1)
                loss_domain = F.cross_entropy(logits_flat, d_rep.long())

        # recon
        loss_recon = torch.tensor(0.0, device=device, dtype=dtype)
        if x_recon_tok is not None:
            mask = model._make_recon_mask(xb, model.recon_mask_ratio)
            if mask is None:
                loss_recon = F.mse_loss(x_recon_tok, xb)
            else:
                diff = (x_recon_tok - xb)[mask]
                loss_recon = torch.mean(diff * diff)

        # loss = loss_weak + loss_domain + model.recon_weight * loss_recon
        lambda_now = (model.grl.lambd if (model.stage == 2 and getattr(model, "grl", None) is not None) else 0.0)
        loss = loss_weak + lambda_now * loss_domain + model.recon_weight * loss_recon



        sum_loss  += float(loss.item())       * B
        sum_weak  += float(loss_weak.item())  * B
        sum_dom   += float(loss_domain.item())* B
        sum_recon += float(loss_recon.item()) * B

        del xb, yb, db, p_spot, logits_dom_tok, x_recon_tok, z_tok
    if device == 'cuda':
        torch.cuda.empty_cache()

    if total_N == 0:
        raise RuntimeError("OOS loader produced no batches.")
    return {
        "val_loss_oos":    sum_loss / total_N,
        "weak_mse_oos":    sum_weak / total_N,
        "domain_cdan_oos": sum_dom / total_N,
        "recon_tok_oos":   sum_recon / total_N,
        "n_samples":       int(total_N),
    }


class MetricTracker(pl.Callback):
    def __init__(self):
        self.collection = []
    def on_train_epoch_end(self, trainer, pl_module):
        self.collection.append(dict(trainer.logged_metrics))
    def clean(self):
        pass


class FeedForward(nn.Module):
    def __init__(self, n_inp, n_out, activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual
    def forward(self, x, indices=None):
        # Works for [..., n_inp] -> [..., n_out]
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta
    def forward(self, x):
        return self.activation(x) + self.beta


# ---------- Gradient Reversal ----------
class _GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GradReverse(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return _GradReverseFn.apply(x, self.lambd)


# ---------- Model (per-token everything) ----------
class MultiTaskModel(pl.LightningModule):
    """
    Per-token weak supervision + per-token CDAN + per-token reconstruction.
    Stage 1: recon only; Stage 2: weak + CDAN (optional entropy conditioning).
    """
    def __init__(
        self,
        lr: float,
        n_inp: int,
        n_ctypes: int,
        n_domains: int = 0,
        adv_lambda: float = 0.0,          # CDAN GRL max
        recon_weight: float = 0.0,
        recon_mask_ratio: float = 0.3,
        token_dim: int = 256,
        entropy_cond: bool = False,
        stage: int = 2,                   # 1 or 2
        freeze_encoder_n: int = 0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.n_inp = n_inp
        self.n_ctypes = n_ctypes
        self.n_domains = n_domains
        self.adv_lambda_max = float(adv_lambda)
        self.recon_weight = float(recon_weight)
        self.recon_mask_ratio = float(recon_mask_ratio)
        self.token_dim = token_dim
        self.entropy_cond = bool(entropy_cond)
        self.stage = int(stage)
        self.freeze_encoder_n = int(freeze_encoder_n)

        # Token encoder (unchanged: 4 FF blocks)
        self.net_lat = nn.Sequential(
            FeedForward(n_inp, token_dim),
            FeedForward(token_dim, token_dim),
            FeedForward(token_dim, token_dim),
            FeedForward(token_dim, token_dim),
        )

        # Cell-type head (per-token)
        self.ct_head_tok = nn.Linear(token_dim, n_ctypes)

        # CDAN domain head on outer-product features z ⊗ p
        if (self.n_domains or 0) > 0:
            self.grl = GradReverse(0.0)  # lambd is ramped each epoch; stays 0.0 in stage 1
            self.domain_head_cdan = nn.Sequential(
                nn.Linear(token_dim * n_ctypes, max(256, token_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(max(256, token_dim), n_domains),
            )
        else:
            self.grl = None
            self.domain_head_cdan = None

        # Reconstruction head (optional), predicts original x from z_tok
        self.recon_head_tok = None
        if self.recon_weight > 0:
            self.recon_head_tok = nn.Sequential(
                nn.Linear(token_dim, token_dim),
                nn.ReLU(inplace=True),
                nn.Linear(token_dim, n_inp),
            )

        self._apply_stage_freeze()

    # ---- utilities ----
    def set_stage(self, stage: int, freeze_encoder_n: int = None):
        self.stage = int(stage)
        if freeze_encoder_n is not None:
            self.freeze_encoder_n = int(freeze_encoder_n)
        self._apply_stage_freeze()

    def _apply_stage_freeze(self):
        # Stage 1: freeze ct/domain heads; encoder trainable
        if self.stage == 1:
            for p in self.ct_head_tok.parameters(): p.requires_grad = False
            if self.domain_head_cdan is not None:
                for p in self.domain_head_cdan.parameters(): p.requires_grad = False
            for m in self.net_lat.modules():
                for p in getattr(m, 'parameters', lambda: [])(): p.requires_grad = True
        elif self.stage == 2:
            for p in self.ct_head_tok.parameters(): p.requires_grad = True
            if self.domain_head_cdan is not None:
                for p in self.domain_head_cdan.parameters(): p.requires_grad = True
            ff_idx = 0
            for m in self.net_lat:
                if isinstance(m, FeedForward):
                    ff_idx += 1
                    freeze_now = (ff_idx <= max(0, self.freeze_encoder_n))
                    for p in m.parameters(): p.requires_grad = not freeze_now
                else:
                    for p in m.parameters(): p.requires_grad = True
        else:
            raise ValueError('stage must be 1 or 2')

    @staticmethod
    def _entropy(p):
        eps = 1e-5
        return -(p * (p + eps).log()).sum(dim=-1)

    def _cdan_features(self, z_tok, p_tok):
        B, T, D = z_tok.shape
        K = p_tok.shape[-1]
        zf = z_tok.view(B*T, D)
        pf = p_tok.view(B*T, K)
        outer = torch.bmm(pf.unsqueeze(2), zf.unsqueeze(1))  # [BT,K,D]
        outer = outer / math.sqrt(D)                         # <— scale
        return outer.view(B, T, K*D)

    # keep for downstream export
    def inp_to_lat(self, x):
        return self.net_lat(x)  


    @staticmethod
    def _make_recon_mask(x, ratio: float):
        if ratio <= 0.0: return None
        B, T, C = x.shape
        return (torch.rand(B, T, C, device=x.device) < ratio)

    def forward(self, x):
        z_tok = self.inp_to_lat(x)                        # [B, T, token_dim]
        p_spot = None
        logits_dom_tok = None

        if self.stage == 2:
            logits_ct_tok = self.ct_head_tok(z_tok)         # [B,T,K]
            p_tok = torch.softmax(logits_ct_tok, dim=-1)    # [B,T,K]
            p_spot = p_tok.mean(dim=1)                      # [B,K]
            if self.domain_head_cdan is not None:
                cond = self._cdan_features(z_tok, p_tok)    # [B,T,K*D]
                logits_dom_tok = self.domain_head_cdan(self.grl(cond))  # [B,T,n_domains]

        x_recon_tok = None
        if self.recon_head_tok is not None:
            x_recon_tok = self.recon_head_tok(z_tok)

        return p_spot, logits_dom_tok, x_recon_tok, z_tok

    def training_step(self, batch, batch_idx):
        x, y_soft, d_lbl = batch
        x = x.float(); y_soft = y_soft.float()
        p_spot, logits_dom_tok, x_recon_tok, z_tok = self.forward(x)

        # Weak (stage 2 only)
        loss_weak = torch.tensor(0.0, device=self.device)
        if self.stage == 2 and p_spot is not None:
            loss_weak = F.mse_loss(p_spot, y_soft)
            self.log('train_weak_mse', loss_weak, prog_bar=True)

        # CDAN (stage 2 only)
        loss_domain = torch.tensor(0.0, device=self.device)
        if self.stage == 2 and logits_dom_tok is not None and d_lbl is not None:
            B, T, D = logits_dom_tok.shape
            d_valid = (d_lbl >= 0)
            if torch.any(d_valid):
                logits_flat = logits_dom_tok[d_valid].reshape(-1, D)  # [sum(Bv)*T, D]
                d_rep = d_lbl[d_valid].view(-1,1).expand(-1,T).reshape(-1)
                if self.entropy_cond:
                    with torch.no_grad():
                        p_tok_all = torch.softmax(self.ct_head_tok(z_tok), dim=-1)
                        H = self._entropy(p_tok_all)                  # [B,T]
                        w = (1.0 + torch.exp(-H[d_valid].reshape(-1))) # [sum* T]
                        w = w / (w.mean() + 1e-6)
                    ce = F.cross_entropy(logits_flat, d_rep.long(), reduction='none')
                    loss_domain = torch.mean(ce * w)
                else:
                    loss_domain = F.cross_entropy(logits_flat, d_rep.long())
                self.log('train_domain_cdan', loss_domain, prog_bar=True)

        # Recon (both stages if enabled)
        loss_recon = torch.tensor(0.0, device=self.device)
        if x_recon_tok is not None:
            mask = self._make_recon_mask(x, self.recon_mask_ratio)
            if mask is None:
                loss_recon = F.mse_loss(x_recon_tok, x)
            else:
                diff = (x_recon_tok - x)[mask]
                loss_recon = torch.mean(diff * diff)
            self.log('train_recon_tok', loss_recon, prog_bar=True)

        # after computing loss_weak, loss_domain, loss_recon
        lambda_now = (self.grl.lambd if (self.stage == 2 and self.grl is not None) else 0.0)
        loss = loss_weak + lambda_now * loss_domain + self.recon_weight * loss_recon

        # you can still log the *raw* CE for interpretability:
        self.log('train_domain_cdan', loss_domain, prog_bar=True)
        # and optionally the scaled term that truly enters the objective:
        self.log('train_domain_cdan_scaled', lambda_now * loss_domain)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_soft, d_lbl = batch
        x = x.float(); y_soft = y_soft.float()
        p_spot, logits_dom_tok, x_recon_tok, z_tok = self.forward(x)

        loss_weak = torch.tensor(0.0, device=self.device)
        if self.stage == 2 and p_spot is not None:
            loss_weak = F.mse_loss(p_spot, y_soft)

        loss_domain = torch.tensor(0.0, device=self.device)
        if self.stage == 2 and logits_dom_tok is not None and d_lbl is not None:
            B, T, D = logits_dom_tok.shape
            d_valid = (d_lbl >= 0)
            if torch.any(d_valid):
                logits_flat = logits_dom_tok[d_valid].reshape(-1, D)
                d_rep = d_lbl[d_valid].view(-1,1).expand(-1,T).reshape(-1)
                loss_domain = F.cross_entropy(logits_flat, d_rep.long())

        loss_recon = torch.tensor(0.0, device=self.device)
        if x_recon_tok is not None:
            mask = self._make_recon_mask(x, self.recon_mask_ratio)
            if mask is None:
                loss_recon = F.mse_loss(x_recon_tok, x)
            else:
                diff = (x_recon_tok - x)[mask]
                loss_recon = torch.mean(diff * diff)

        # after computing loss_weak, loss_domain, loss_recon
        lambda_now = (self.grl.lambd if (self.stage == 2 and self.grl is not None) else 0.0)
        loss = loss_weak + lambda_now * loss_domain + self.recon_weight * loss_recon

        # you can still log the *raw* CE for interpretability:
        self.log('train_domain_cdan', loss_domain, prog_bar=True)
        # and optionally the scaled term that truly enters the objective:
        self.log('train_domain_cdan_scaled', lambda_now * loss_domain)

        self.log('val_loss', loss, prog_bar=True)
        if self.stage == 2:
            self.log('val_weak_mse', loss_weak)
            if logits_dom_tok is not None: self.log('val_domain_cdan', loss_domain)
        if x_recon_tok is not None: self.log('val_recon_tok', loss_recon)
        return loss

    def configure_optimizers(self):
        # respect stage-based freezing
        # return Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        try:
            return Adam(filter(lambda p: p.requires_grad, self.parameters()),
                        lr=self.lr, fused=True)
        except TypeError:
            return Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    def on_train_epoch_start(self):
        # CDAN schedule: ramp GRL 0 → adv_lambda_max (stage 2 only)
        if self.stage == 2 and self.grl is not None and self.trainer is not None and self.adv_lambda_max > 0:
            p = self.current_epoch / max(1, self.trainer.max_epochs - 1)
            lam = self.adv_lambda_max * (2.0 / (1.0 + math.exp(-10.0*p)) - 1.0)
            self.grl.lambd = float(lam)
            self.log("cdan_lambda_sched", self.grl.lambd, prog_bar=True)
            


# ---------- Dataset & batching (keep tokens; no normalization) ----------
class CachedBatchDataset(Dataset):
    """
    Memory-efficient batched dataset.

    On-disk formats inside `batch_dir`:

    1) Preferred (memmap):
         batch_XXX_x.npy : float16 [N, T, C]
         batch_XXX_y.npy : float32 [N, K]
         batch_XXX_d.npy : int64   [N]  (optional; domain id per spot, -1 allowed)
       Loaded with mmap_mode='r' so workers share the OS page cache.

    2) Legacy (pickle):
         batch_XXX.pkl containing (x:[N,T,C], y:[N,K], d:[N]) or (x, y)

    Notes:
    - Tiny LRU (default=2) avoids closing memmaps while a worker is still using them.
    - Uses cumulative sizes + bisect to avoid building a huge sample_to_batch list.
    """

    def __init__(self, batch_dir: str, max_cached_batches: int = 2):
        super().__init__()
        self.batch_dir = batch_dir
        self.max_cached_batches = max_cached_batches

        # LRU: batch_idx -> (x, y[, d])
        self.batch_cache = {}
        self.cache_order = []

        # ---- Discover files ----
        npy_x = sorted(
            f for f in os.listdir(batch_dir)
            if f.startswith("batch_") and f.endswith("_x.npy")
        )
        if npy_x:
            self.mode = "npy"
            # stems like 'batch_000' (strip '_x.npy')
            self.batch_stems = [f[:-6] for f in npy_x]
            self.batch_files = self.batch_stems[:]  # alias for compatibility
        else:
            self.mode = "pkl"
            self.batch_files = sorted(
                f for f in os.listdir(batch_dir)
                if f.startswith("batch_") and f.endswith(".pkl")
            )
            if not self.batch_files:
                raise RuntimeError(f"No batch files found in {batch_dir}")

        # ---- Lightweight index (sizes/shapes/domain stats) ----
        self.batch_sizes = []
        self.cumulative_sizes = [0]
        self.total_size = 0
        self.has_domain = False
        self.n_domains = 0

        if self.mode == "npy":
            # Probe shapes from first batch (memmap; no RAM copy)
            x0 = np.load(os.path.join(batch_dir, self.batch_stems[0] + "_x.npy"), mmap_mode="r")
            y0 = np.load(os.path.join(batch_dir, self.batch_stems[0] + "_y.npy"), mmap_mode="r")
            self.token_len = int(x0.shape[1])
            self.n_inp     = int(x0.shape[2])
            self.n_out     = int(y0.shape[1])
            del x0, y0

            # Build sizes and domain stats; avoid keeping handles alive
            for i, stem in enumerate(self.batch_stems):
                x = np.load(os.path.join(batch_dir, stem + "_x.npy"), mmap_mode="r")
                bsz = int(x.shape[0])
                del x
                self.batch_sizes.append(bsz)
                self.total_size += bsz
                self.cumulative_sizes.append(self.total_size)

                d_path = os.path.join(batch_dir, stem + "_d.npy")
                if os.path.exists(d_path):
                    self.has_domain = True
                    d = np.load(d_path, mmap_mode="r")
                    if d.size > 0:
                        max_val = int(d.max())
                        if max_val >= 0:
                            self.n_domains = max(self.n_domains, max_val + 1)
                    del d

                if (i + 1) % 32 == 0:  # be nice to the GC on huge sets
                    gc.collect()

        else:
            # Legacy path: we must touch pickles to know shapes/sizes
            with open(os.path.join(batch_dir, self.batch_files[0]), "rb") as f:
                loaded0 = pickle.load(f)
            if len(loaded0) == 2:
                fx, fy = loaded0
            else:
                fx, fy, _ = loaded0
            self.token_len = int(fx.shape[1])
            self.n_inp     = int(fx.shape[2])
            self.n_out     = int(fy.shape[1])
            del loaded0, fx, fy

            for i, bf in enumerate(self.batch_files):
                with open(os.path.join(batch_dir, bf), "rb") as f:
                    loaded = pickle.load(f)
                if len(loaded) == 2:
                    bx, by = loaded
                    bd = None
                else:
                    bx, by, bd = loaded

                bsz = int(bx.shape[0])
                self.batch_sizes.append(bsz)
                self.total_size += bsz
                self.cumulative_sizes.append(self.total_size)

                if bd is not None and len(bd) > 0:
                    self.has_domain = True
                    try:
                        max_val = int(np.max(bd))
                        if max_val >= 0:
                            self.n_domains = max(self.n_domains, max_val + 1)
                    except Exception:
                        pass

                del bx, by, bd, loaded
                if (i + 1) % 8 == 0:
                    gc.collect()

        print(
            f"Dataset ready: {len(self.batch_sizes)} batches, {self.total_size} samples\n"
            f"T={self.token_len}, C={self.n_inp}, K={self.n_out}, "
            f"has_domain={self.has_domain}, n_domains={self.n_domains}"
        )

    # --------------------- Internal helpers ---------------------

    def _evict_until_within_budget(self):
        # IMPORTANT: do not manually close memmaps here—just drop references.
        while len(self.batch_cache) > self.max_cached_batches:
            oldest = self.cache_order.pop(0)
            self.batch_cache.pop(oldest, None)

    def _load_batch(self, batch_idx):
        # LRU hit
        if batch_idx in self.batch_cache:
            try:
                self.cache_order.remove(batch_idx)
            except ValueError:
                pass
            self.cache_order.append(batch_idx)
            return self.batch_cache[batch_idx]

        # Miss → load
        if self.mode == "npy":
            stem = self.batch_stems[batch_idx]
            x = np.load(os.path.join(self.batch_dir, stem + "_x.npy"), mmap_mode="r")
            y = np.load(os.path.join(self.batch_dir, stem + "_y.npy"), mmap_mode="r")
            d_path = os.path.join(self.batch_dir, stem + "_d.npy")
            d = np.load(d_path, mmap_mode="r") if os.path.exists(d_path) else None
            entry = (x, y, d) if d is not None else (x, y)
        else:
            with open(os.path.join(self.batch_dir, self.batch_files[batch_idx]), "rb") as f:
                entry = pickle.load(f)  # (x,y) or (x,y,d)

        self.batch_cache[batch_idx] = entry
        self.cache_order.append(batch_idx)
        self._evict_until_within_budget()
        return entry

    # --------------------- PyTorch Dataset API ---------------------

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # support negative indexing
        if idx < 0:
            idx += self.total_size
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range 0..{self.total_size-1}")

        # find which batch this global index falls into
        # cumulative_sizes = [0, b1, b1+b2, ..., total]
        batch_idx = bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[batch_idx]

        loaded = self._load_batch(batch_idx)
        if len(loaded) == 2:
            bx, by = loaded
            return bx[local_idx], by[local_idx], -1
        else:
            bx, by, bd = loaded
            d_val = int(bd[local_idx]) if bd is not None else -1
            return bx[local_idx], by[local_idx], d_val

    # --------------------- Cleanup ---------------------

    def close(self):
        """
        Optional: call at end-of-job to drop references eagerly.
        (We do not manually close memmaps while loaders may still read.)
        """
        self.batch_cache.clear()
        self.cache_order.clear()
        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

# ---------- Data helpers ----------
def get_disk(img, ij, radius):
    i, j = ij
    patch = img[i-radius:i+radius, j-radius:j+radius]
    disk_mask = get_disk_mask(radius)
    patch[~disk_mask] = 0.0
    return patch

def get_patches_tokens(img, locs, mask):
    """
    Returns per-spot token arrays by masking disk region:
      img: [H, W, C]
      mask: [2r, 2r] boolean disk
    Output: [N_spots, T, C]
    """
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # 2 x 2
    x_list = []
    for s in locs:
        patch = img[s[0]+r[0][0]:s[0]+r[0][1], s[1]+r[1][0]:s[1]+r[1][1]]  # [2r,2r,C]
        # select only disk pixels -> [T, C]
        x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)  # [N_spots, T, C]
    return x_list

def predict_single_lat(model, x):   # x can be numpy or tensor
    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        x_t = torch.as_tensor(x, device=device, dtype=model_dtype)
        z_tok = model.inp_to_lat(x_t)   # [N, T, token_dim]
        return z_tok.detach().cpu().numpy()


def get_data(prefix):
    gene_names = read_lines(f'{prefix}anno-names.txt')  # cell-type names
    cnts = get_gene_counts(prefix)                      # proportions per spot (DF, N_spots × K)
    cnts = cnts[gene_names]                             # keep correct column order

    embs = load_pickle(f'{prefix}embeddings-hist.pickle')  # [H,W,C]
    locs = get_locs(prefix, target_shape=embs.shape[:2])   # [N_spots, 2] with "spot" as index or column
    return embs, cnts, locs

def filter_and_save_shared_spots(prefix, overwrite=True, make_backup=True):
    """
    Restrict anno_matrix.tsv and locs.tsv to shared spots only.
    If overwrite=True: write back to the SAME filenames (back up originals).
    Ensures both files are sorted by 'spot' so row orders match.
    """
    anno_file = os.path.join(prefix, "anno_matrix.tsv")
    locs_file = os.path.join(prefix, "locs.tsv")

    anno = pd.read_csv(anno_file, sep="\t")
    locs = pd.read_csv(locs_file, sep="\t")

    if "spot" not in anno.columns or "spot" not in locs.columns:
        raise ValueError(f"Missing 'spot' column in {prefix} (anno_matrix.tsv or locs.tsv).")

    shared_spots = set(anno["spot"]).intersection(set(locs["spot"]))
    if not shared_spots:
        raise ValueError(f"No shared spots found in {prefix}")

    # Filter and sort BOTH by 'spot' to guarantee identical row order
    anno_shared = anno[anno["spot"].isin(shared_spots)].sort_values("spot")
    locs_shared = locs[locs["spot"].isin(shared_spots)].sort_values("spot")

    if overwrite:
        if make_backup:
            # keep one-time backups; if already exist, don't overwrite
            anno_bak = anno_file + ".bak"
            locs_bak = locs_file + ".bak"
            if not os.path.exists(anno_bak):
                shutil.copyfile(anno_file, anno_bak)
            if not os.path.exists(locs_bak):
                shutil.copyfile(locs_file, locs_bak)
        # overwrite originals so your existing loaders pick them up
        anno_shared.to_csv(anno_file, sep="\t", index=False)
        locs_shared.to_csv(locs_file, sep="\t", index=False)
        print(f"[align-spots] Kept {len(shared_spots)} shared spots. Overwrote anno_matrix.tsv and locs.tsv in {prefix}")
    else:
        # optional branch if you ever want side-by-side files
        anno_shared.to_csv(os.path.join(prefix, "anno_matrix_shared.tsv"), sep="\t", index=False)
        locs_shared.to_csv(os.path.join(prefix, "locs_shared.tsv"), sep="\t", index=False)
        print(f"[align-spots] Kept {len(shared_spots)} shared spots. Wrote *_shared.tsv in {prefix}")

# ---------- Domain mapping ----------
def _auto_domain_map_from_samples(sample_dirs):
    names = [os.path.basename(os.path.normpath(d)) for d in sample_dirs]
    uniq = sorted(set(names))
    name_to_id = {n: i for i, n in enumerate(uniq)}
    return name_to_id, uniq

def _load_domain_map(domain_map_file):
    if domain_map_file is None or (not os.path.exists(domain_map_file)):
        return {}, []
    name_to_domstr = {}
    with open(domain_map_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            k, v = line.split('\t')
            name_to_domstr[k] = v
    dom_names = sorted(set(name_to_domstr.values()))
    dom_to_id = {d: i for i, d in enumerate(dom_names)}
    name_to_id = {k: dom_to_id[v] for k, v in name_to_domstr.items()}
    return name_to_id, dom_names

# ---------- Sample completeness checks ----------
def _read_radius(sample_dir):
    try:
        return int(read_string(os.path.join(sample_dir, 'radius.txt'))) // 16
    except Exception:
        return None

def check_sample_complete(sample_dir, canonical_radius=None):
    """
    Returns (is_ok: bool, reason: str | None).
    We try to actually touch the data using existing loaders so any hidden
    parsing issues are caught up front.
    - Required: radius.txt, anno-names.txt, embeddings-hist.pickle, locs (via get_locs),
                counts (via get_gene_counts)
    - Optional: canonical_radius to enforce same radius across samples
    """
    # 1) cheap file presence checks for common paths used by get_data()
    req_paths = [
        os.path.join(sample_dir, 'anno-names.txt'),
        os.path.join(sample_dir, 'embeddings-hist.pickle'),
        # get_locs/get_gene_counts do their own path logic; we still sanity check common names:
        os.path.join(sample_dir, 'locs.tsv'),
        os.path.join(sample_dir, 'anno_matrix.tsv'),
    ]
    for p in req_paths:
        if not os.path.exists(p):
            return False, f"missing {os.path.basename(p)}"

    # 2) radius check (and canonical enforcement)
    r = _read_radius(sample_dir)
    if r is None:
        return False, "missing/invalid radius.txt"
    if canonical_radius is not None and r != canonical_radius:
        return False, f"radius mismatch (got {r}, need {canonical_radius})"

    # 3) light load to catch parsing / shape mismatches
    try:
        embs, cnts, locs = get_data(sample_dir)  # may raise if anything is off
    except Exception as e:
        return False, f"load error: {e}"

    # 4) shape sanity: per-spot rows must match
    n_spots = len(cnts)
    if not isinstance(locs, np.ndarray):
        return False, "locs is not a numpy array"
    if locs.ndim != 2 or locs.shape[1] != 2:
        return False, f"locs has wrong shape {locs.shape}"
    if n_spots != locs.shape[0]:
        return False, f"spot mismatch counts({n_spots}) != locs({locs.shape[0]})"

    # 5) basic emb shape
    if not isinstance(embs, np.ndarray) or embs.ndim != 3:
        return False, f"embeddings invalid shape {getattr(embs, 'shape', None)}"

    # 6) can we extract tokens for at least one spot?
    try:
        mask = get_disk_mask(r)
        if mask.ndim != 2:
            return False, "disk mask not 2D"
        # probe the first spot only
        _ = get_patches_tokens(embs, locs[:1], mask)  # [1, T, C] if ok
    except Exception as e:
        return False, f"tokenization error: {e}"

    return True, None

# ---------- Prepare & save batches (keep tokens!) ----------
def prepare_and_save_batches(sample_dirs, radius, output_dir, samples_per_batch=3, domain_map_file=None):
    """
    Saves batches as (x_tokens, y_soft, domain_id).
      x_tokens: [N_spots, T, C]  (no flatten)
      y_soft:   [N_spots, K]
      domain_id:[N_spots]
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(sample_dirs)} samples in batches of {samples_per_batch}")

    # domain mapping: prefer file; else auto per-WSI
    name_to_domain_id_file, dom_names_file = _load_domain_map(domain_map_file)
    if name_to_domain_id_file:
        name_to_domain_id = name_to_domain_id_file
        dom_names = dom_names_file
        print(f"[Domain] Loaded from file: {len(dom_names)} domains")
    else:
        name_to_domain_id, dom_names = _auto_domain_map_from_samples(sample_dirs)
        print(f"[Domain] Auto per-WSI: {len(dom_names)} domains")

    batch_count = 0
    for i in range(0, len(sample_dirs), samples_per_batch):
        batch_dirs = sample_dirs[i:i+samples_per_batch]
        print(f"\nProcessing batch {batch_count + 1} ({len(batch_dirs)} samples)...")

        x_batch, y_batch, d_batch = [], [], []
        for j, d in enumerate(batch_dirs):
            try:
                sample_name = os.path.basename(os.path.normpath(d))
                print(f"  Loading sample {i+j+1}/{len(sample_dirs)}: {sample_name}")

                embs, cnts, locs = get_data(d)  # embs: [H,W,C], cnts: DF (N_spots,K)
                y_soft = cnts.to_numpy().astype(np.float32)

                mask = get_disk_mask(radius)    # [2r,2r] boolean disk
                x_tokens = get_patches_tokens(embs, locs, mask)  # [N_spots, T, C]

                # basic validity: finite tokens across features
                valid = np.isfinite(x_tokens).all(-1).all(-1)  # [N_spots]
                if valid.sum() > 0:
                    xv = x_tokens[valid].astype(np.float16)
                    yv = y_soft[valid]
                    x_batch.append(xv)
                    y_batch.append(yv)
                    dom_id = name_to_domain_id.get(sample_name, -1)
                    d_batch.append(np.full((xv.shape[0],), dom_id, dtype=np.int64))
                    print(f"    Added {valid.sum()} spots (domain_id={dom_id})")
                else:
                    print(f"    No valid spots found")

                del embs, cnts, locs, x_tokens, valid
                gc.collect()
            except Exception as e:
                print(f"    ERROR loading sample: {e}")
                continue

        if x_batch:
            try:
                x_batch_concat = np.concatenate(x_batch)       # [N, T, C]
                y_batch_concat = np.concatenate(y_batch)       # [N, K]
                d_batch_concat = np.concatenate(d_batch)       # [N]

                batch_file = os.path.join(output_dir, f'batch_{batch_count:03d}.pkl')
                # --- BEFORE ---
                # with open(batch_file, 'wb') as f:
                #     pickle.dump((x_batch_concat, y_batch_concat, d_batch_concat), f)

                # --- AFTER: write 3 .npy files per batch (mmap-friendly) ---
                bx = os.path.join(output_dir, f"batch_{batch_count:03d}_x.npy")
                by = os.path.join(output_dir, f"batch_{batch_count:03d}_y.npy")
                bd = os.path.join(output_dir, f"batch_{batch_count:03d}_d.npy")
                np.save(bx, x_batch_concat)   # float16
                np.save(by, y_batch_concat)   # float32
                np.save(bd, d_batch_concat)   # int64
                print(f"  Saved batch {batch_count}: {x_batch_concat.shape[0]} spots -> {bx}, {by}, {bd}")
                batch_count += 1


                del x_batch_concat, y_batch_concat, d_batch_concat
            except Exception as e:
                print(f"  ERROR saving batch: {e}")

        del x_batch, y_batch, d_batch
        gc.collect()

    print(f"\nCompleted: {batch_count} batches saved to {output_dir}")
    return batch_count


def _read_all_domains_fast(dataset: CachedBatchDataset):
    # Always return a real RAM array, never a memmap view.
    dom_ids = []
    if isinstance(dataset, CachedBatchDataset) and getattr(dataset, "mode", None) == "npy":
        for stem in dataset.batch_stems:
            d_path = os.path.join(dataset.batch_dir, stem + "_d.npy")
            if os.path.exists(d_path):
                d_mm = np.load(d_path, mmap_mode='r')
                dom_ids.append(np.array(d_mm, copy=True))   # <— force copy!
                del d_mm
        return np.concatenate(dom_ids) if dom_ids else np.array([], dtype=np.int64)
    elif isinstance(dataset, CachedBatchDataset) and getattr(dataset, "mode", None) == "pkl":
        for bf in dataset.batch_files:
            with open(os.path.join(dataset.batch_dir, bf), 'rb') as f:
                loaded = pickle.load(f)
            if len(loaded) == 3 and loaded[2] is not None:
                dom_ids.append(np.array(loaded[2], copy=True))
        return np.concatenate(dom_ids) if dom_ids else np.array([], dtype=np.int64)
    else:
        return np.array([], dtype=np.int64)

def get_dom_ids_for_dataset(ds):
    if isinstance(ds, torch.utils.data.Subset):
        base = ds.dataset
        idxs = np.asarray(ds.indices, dtype=np.int64)
        dom_all = _read_all_domains_fast(base)
        if dom_all.size == 0:
            return np.full(len(idxs), -1, dtype=np.int64)
        return dom_all[idxs]
    elif isinstance(ds, CachedBatchDataset):
        dom_all = _read_all_domains_fast(ds)
        return dom_all if dom_all.size else np.full(len(ds), -1, dtype=np.int64)
    else:
        # slow but safe fallback
        out = np.empty(len(ds), dtype=np.int64)
        for i in range(len(ds)):
            _, _, d = ds[i]
            out[i] = int(d)
        return out


def auto_batch_size(dataset, target_tokens=300000, hard_cap=128):
    T = int(dataset.token_len)
    bs = max(1, target_tokens // max(1, T))
    return int(min(hard_cap, bs))

# ---------- Train wrappers ----------
def create_train_val_split(dataset, val_ratio=0.1, random_seed=42):
    np.random.seed(random_seed)
    dataset_size = len(dataset)
    indices = np.random.permutation(dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    return train_dataset, val_dataset

def train_model(
    dataset,
    batch_size,
    epochs,
    model=None,
    model_class=None,
    model_kwargs={},
    device='cuda',
    val_dataset=None,
    save_every_n_epochs=10,
    prefix="./",
    patience=0,                 # 0 disables EarlyStopping
    monitor_metric="val_loss",  # what to track for best/early stopping
    resume_ckpt=None ,           # optional path to resume training
    extra_callbacks=None
):
    """
    Train with domain-balanced mini-batches when domain labels are available.

    Balancing logic:
      - If the (train) dataset returns domain ids (d >= 0), we compute per-sample
        weights ~ 1/freq(domain) on the actual training subset and use a
        WeightedRandomSampler so each mini-batch tends to contain both domains.
      - If no valid domains are present, we fall back to shuffle=True.
      
    """
    if model is None:
        model = model_class(**model_kwargs)


    # num_workers = max(4, os.cpu_count() // 6)
    num_workers = DEFAULT_NUM_WORKERS

    # keep each worker's dataset cache tiny
    if hasattr(dataset, "max_cached_batches"):
        if num_workers > 0:
            dataset.max_cached_batches = 2

    # ---------- Build balanced sampler for TRAIN ----------
    # We'll read domain ids from *the dataset passed in* (often a Subset),
    # so the statistics reflect the training split only.
    use_balanced_sampler = False
    sampler = None

    try:
        n_train = len(dataset)
        # Pull domain ids once (cheap if batches are cached)
        dom_ids = get_dom_ids_for_dataset(dataset)



        valid_dom = [d for d in dom_ids if d >= 0]
        if len(valid_dom) > 0:
            max_dom = max(valid_dom)
            # require at least 2 distinct domains to bother balancing
            if len(set(valid_dom)) >= 2:
                counts = torch.bincount(torch.tensor(valid_dom, dtype=torch.long),
                                        minlength=max_dom + 1).float()
                class_w = 1.0 / torch.clamp(counts, min=1.0)
                # per-sample weights (unknown domain -> weight 1.0)
                sample_w = torch.tensor(
                    [class_w[d].item() if (d >= 0 and d < len(class_w)) else 1.0 for d in dom_ids],
                    dtype=torch.float
                )
                sampler = WeightedRandomSampler(sample_w, num_samples=n_train, replacement=True)
                use_balanced_sampler = True
                # helpful summary
                with torch.no_grad():
                    msg = ", ".join([f"d{j}={int(counts[j].item())}" for j in range(len(counts))])
                print(f"[train] domain balancing ON | counts: {msg}")
            else:
                print("[train] domain balancing OFF (only one domain present in the train split).")
        else:
            print("[train] domain balancing OFF (no valid domain labels in train split).")
    except Exception as e:
        # Never fail training because of balancing
        print(f"[train] domain balancing skipped due to: {e}")

    # Train loader (balanced if possible)
    if use_balanced_sampler and sampler is not None:
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,               # <— balanced sampling
            shuffle=False,                 # must be False when sampler is used
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True, 
            prefetch_factor=DEFAULT_PREFETCH_FACTOR
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,                  # standard shuffle
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True, 
            prefetch_factor=DEFAULT_PREFETCH_FACTOR
        )

    # Val loader
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True, 
            prefetch_factor=DEFAULT_PREFETCH_FACTOR
        )

    # ----- Loggers -----
    logs_dir = os.path.join(prefix, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    tb_logger  = TensorBoardLogger(save_dir=logs_dir, name="tb")
    csv_logger = CSVLogger(save_dir=logs_dir, name="csv")

    # ----- Callbacks -----
    callbacks = [MetricTracker(), RichProgressBar(), LearningRateMonitor(logging_interval='epoch')]

    # Save every N epochs
    ckpt_dir = os.path.join(prefix, "epoch_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=save_every_n_epochs,
        save_top_k=-1
    ))

    # Save top-K best (needs val set)
    if val_dataset is not None:
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=3
        ))

    # Optional EarlyStopping
    if patience and val_dataset is not None:
        callbacks.append(EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=patience,
            min_delta=0.0
        ))
    # Extra user callbacks
    if extra_callbacks:
            callbacks.extend(extra_callbacks)
    accelerator = {'cuda': 'gpu', 'cpu': 'cpu'}[device]
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
        gradient_clip_val= 0.0,               # <— safe clip to tame occasional spikes
        # gradient_clip_algorithm="norm"
    )

    model.train()
    t0 = time()
    # if hasattr(torch, "compile"):
    #     try:
    #         import torch._inductor.config as inductor_config
    #         inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    #         inductor_config.triton.cudagraph_dynamic_shape_warn_limit = 0  # silence
    #     except Exception:
    #         pass
    #     model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=resume_ckpt)
    print(f"Training completed in {int(time() - t0)} seconds")

    tracker = callbacks[0]
    tracker.clean()
    history = tracker.collection
    return model, history, trainer


def get_model(model_class, model_kwargs, dataset, prefix, epochs=None, device='cuda', load_saved=False, **kwargs):
    checkpoint_file = prefix + 'model.ckpt'
    history_file = prefix + 'history.pickle'
    if load_saved and os.path.exists(checkpoint_file):
        model = model_class.load_from_checkpoint(checkpoint_file)
        print(f'Model loaded from {checkpoint_file}')
        history = load_pickle(history_file)
    else:
        model = None
        history = []
    extra_callbacks = kwargs.pop("extra_callbacks", None) 
    if (epochs is not None) and (epochs > 0):
        model, hist, trainer = train_model(
            model=model, model_class=model_class, model_kwargs=model_kwargs,
            dataset=dataset, epochs=epochs, device=device, prefix=prefix,
            **kwargs,                                   # no extra_callbacks here
            extra_callbacks=extra_callbacks            # pass exactly once
        )
        trainer.save_checkpoint(checkpoint_file)
        print(f'Model saved to {checkpoint_file}')
        history += hist
        save_pickle(history, history_file)
        print(f'History saved to {history_file}')
    return model


def get_model_batched(batch_dir, prefix, batch_size, epochs, lr,
                      load_saved=False, device='cuda', val_ratio=0.1,
                      save_every_n_epochs=10,
                      n_domains: int = 0,
                      adv_lambda: float = 0.0,
                      recon_weight: float = 0.0,
                      recon_mask_ratio: float = 0.3,
                      token_dim: int = 256,
                      monitor_metric='val_loss', patience=0, resume_ckpt=None,
                      # --- new ---
                      entropy_cond: bool=False,
                      two_stage: bool=False,
                      epochs1: int=0,
                      epochs2: int=0,
                      freeze_encoder_n: int=0,
                      stage2_lr: float=None,
                      stage1_prefix_suffix: str='stage1/',
                      stage2_prefix_suffix: str='stage2/', 
                      # --- OOS monitor params (NEW) ---
                      oos_sample: str = None,
                      oos_tmpdir: str = None,
                      oos_batch_size: int = 0):
    print(f'Loading batched dataset from: {batch_dir}')
    dataset = CachedBatchDataset(batch_dir, max_cached_batches=2)
    print(f'Dataset loaded: {len(dataset)} samples, T={dataset.token_len}, C={dataset.n_inp}, K={dataset.n_out}')

    train_dataset, val_dataset = create_train_val_split(dataset, val_ratio=val_ratio)
    print(f'Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')

    n_domains_eff = n_domains if n_domains > 0 else (dataset.n_domains if dataset.has_domain else 0)
    print(f'Using n_domains={n_domains_eff} (CDAN {"enabled" if (n_domains_eff>0 and adv_lambda>0) else "disabled"})')

    # ---------- single-stage (back-compat) ----------
    if not two_stage:
        extra_cbs = []
        if oos_sample:
            extra_cbs.append(OOSMonitor(
                sample_dir=oos_sample,
                prefix=prefix,                 # per-state folder, e.g. .../states/00/
                tmp_dir=oos_tmpdir,
                device=device,
                batch_size=oos_batch_size
            ))

        
        model = get_model(
            model_class=MultiTaskModel,
            model_kwargs=dict(
                n_inp=dataset.n_inp,
                n_ctypes=dataset.n_out,
                n_domains=n_domains_eff,
                lr=lr,
                adv_lambda=adv_lambda,
                recon_weight=recon_weight,
                recon_mask_ratio=recon_mask_ratio,
                token_dim=token_dim,
                entropy_cond=entropy_cond,
                stage=2,
                freeze_encoder_n=freeze_encoder_n
            ),
            dataset=train_dataset,
            prefix=prefix,
            epochs=epochs,
            load_saved=load_saved,
            device=device,
            batch_size=batch_size,
            val_dataset=val_dataset,
            save_every_n_epochs=save_every_n_epochs,
            monitor_metric=monitor_metric,
            patience=patience,
            resume_ckpt=resume_ckpt, 
            extra_callbacks=extra_cbs
        )
        model.eval()
        if device == 'cuda': torch.cuda.empty_cache()
        return model, dataset

    # ---------- two-stage ----------
    # Stage 1: recon-only pretrain (CDAN disabled; ct/domain heads frozen)
    stage1_prefix = os.path.join(prefix, stage1_prefix_suffix)
    os.makedirs(stage1_prefix, exist_ok=True)

    model_stage1 = MultiTaskModel(
        n_inp=dataset.n_inp, n_ctypes=dataset.n_out, n_domains=n_domains_eff,
        lr=lr, adv_lambda=0.0,                    # adversary OFF
        recon_weight=recon_weight, recon_mask_ratio=recon_mask_ratio,
        token_dim=token_dim, entropy_cond=False,
        stage=1, freeze_encoder_n=0
    )
    extra_cbs = []
    if oos_sample:
        extra_cbs.append(OOSMonitor(
            sample_dir=oos_sample,
            prefix=stage1_prefix,          # or stage2_prefix for stage2-only curves
            tmp_dir=oos_tmpdir,
            device=device,
            batch_size=oos_batch_size
        ))
    model_stage1, hist1, trainer1 = train_model(
        model=model_stage1,
        dataset=train_dataset, epochs=epochs1, device=device,
        prefix=stage1_prefix, batch_size=batch_size,
        val_dataset=val_dataset, save_every_n_epochs=save_every_n_epochs,
        monitor_metric='val_loss', patience=patience, resume_ckpt=None,
        extra_callbacks=extra_cbs
    )
    ckpt1 = os.path.join(stage1_prefix, 'model_stage1.ckpt')
    trainer1.save_checkpoint(ckpt1)
    print(f'[two-stage] Stage 1 checkpoint saved to {ckpt1}')

    # Stage 2: enable weak + CDAN; optional LR override and partial freeze
    stage2_prefix = os.path.join(prefix, stage2_prefix_suffix)
    os.makedirs(stage2_prefix, exist_ok=True)

    model_stage1.set_stage(2, freeze_encoder_n=freeze_encoder_n)
    model_stage2 = model_stage1
    model_stage2.lr = (stage2_lr if stage2_lr is not None else lr)
    model_stage2.adv_lambda_max = float(adv_lambda)
    model_stage2.entropy_cond = bool(entropy_cond)

    extra_cbs2 = []
    if oos_sample:
        extra_cbs2.append(OOSMonitor(
            sample_dir=oos_sample,
            prefix=stage2_prefix,
            tmp_dir=oos_tmpdir,
            device=device,
            batch_size=oos_batch_size
        ))
    model_stage2, hist2, trainer2 = train_model(
        model=model_stage2,
        dataset=train_dataset, epochs=epochs2, device=device,
        prefix=stage2_prefix, batch_size=batch_size,
        val_dataset=val_dataset, save_every_n_epochs=save_every_n_epochs,
        monitor_metric=monitor_metric, patience=patience, resume_ckpt=None, 
        extra_callbacks=extra_cbs2
    )

    final_ckpt = os.path.join(prefix, 'model.ckpt')
    trainer2.save_checkpoint(final_ckpt)
    print(f'[two-stage] Final model saved to {final_ckpt}')

    model_stage2.eval()
    if device == 'cuda': torch.cuda.empty_cache()
    return model_stage2, dataset


# ---------- Prediction (save token-level latents; pooled as needed) ----------
def predict_batched(model_states, batch_dir, prefix, device='cuda'):
    """
    Exports token-level latents (median across states).
    Saves: prefix + 'embeddings-gene.pickle' as dict(cls=[D, N, T])
           where cls = z_tok^T format preserved from your previous pipeline.
    """
    model_states = [mod.to(device) for mod in model_states]

    dataset = CachedBatchDataset(batch_dir)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    z_all = []
    print("Predicting token-level latent features...")
    model0 = model_states[0]
    model_dtype = next(model0.parameters()).dtype
    device = next(model0.parameters()).device

    with torch.no_grad():
        for batch_idx, (x_batch, _, __) in enumerate(dataloader):
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
            x_batch = x_batch.to(device=device, dtype=model_dtype)  # <<< key cast

            z_batch_states = []
            for model in model_states:
                z_batch = model.inp_to_lat(x_batch)  # [B, T, D]
                z_batch_states.append(z_batch.detach().cpu())

            z_batch_median = torch.median(torch.stack(z_batch_states, dim=0), dim=0).values  # [B, T, D]
            z_all.append(z_batch_median.numpy())
            del z_batch_states, z_batch_median
            gc.collect()


    z_point = np.concatenate(z_all, axis=0)   # [N, T, D]
    # keep same transpose convention as before:
    z_dict = dict(cls=z_point.transpose(2, 0, 1))  # [D, N, T]
    save_pickle(z_dict, prefix+'embeddings-gene.pickle')
    print(f"Saved latent embeddings: {z_point.shape} -> {prefix+'embeddings-gene.pickle'}")
    del z_point, z_all
    gc.collect()


# ---------- CLI ----------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--n-states', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--load-saved', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--samples-per-batch', type=int, default=3)
    parser.add_argument('--save-every-n-epochs', type=int, default=10)

    # Domains / DANN
    parser.add_argument('--domain-map', type=str, default=None, help='Optional TSV: sample_name\\tdomain_string')
    parser.add_argument('--n-domains', type=int, default=0, help='Overrides detected domains if >0')
    parser.add_argument('--dann-lambda', type=float, default=0.0, help='GRL strength (e.g., 0.1)')

    # Recon
    parser.add_argument('--recon-weight', type=float, default=0.0)
    parser.add_argument('--recon-mask-ratio', type=float, default=0.3)

    # Token dim
    parser.add_argument('--token-dim', type=int, default=256)

    # Model checkpointing
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 disables)')
    parser.add_argument('--monitor-metric', type=str, default='val_loss', help='Metric to monitor for best/ES')
    parser.add_argument('--resume-ckpt', type=str, default=None, help='Path to a checkpoint to resume from')

    # exclude samples
    # In get_args(), add these:
    parser.add_argument('--exclude-samples', type=str, default="",
        help="Comma-separated sample folder names to exclude (e.g., 'P001,P123').")
    parser.add_argument('--exclude-file', type=str, default=None,
        help="Optional text file with one sample folder name per line to exclude (# comments allowed).")
    
    # ---- Include samples (new) ----
    parser.add_argument('--include-samples', type=str, default="",
        help="Comma-separated sample folder names to include exclusively (e.g., 'P001,P123'). "
             "If set, only these samples are considered before exclusions.")
    parser.add_argument('--include-file', type=str, default=None,
        help="Optional text file with one sample folder name per line to include (supports '#' comments).")

    # spots align
    parser.add_argument('--align-spots', action='store_true',
    help="If set, automatically restrict to the intersection of spots between anno_matrix.tsv and locs.tsv")

    # in get_args()
    parser.add_argument('--force-rebuild-batches', action='store_true',
    help='Ignore existing batches and rebuild from raw samples')


    # ---- Two-stage / CDAN ----
    parser.add_argument('--two-stage', action='store_true',
        help='Enable two-stage training (stage1: recon-only; stage2: weak+CDAN)')
    parser.add_argument('--epochs1', type=int, default=0,
        help='Stage 1 epochs (reconstruction-only)')
    parser.add_argument('--epochs2', type=int, default=0,
        help='Stage 2 epochs (weak supervision + CDAN)')
    parser.add_argument('--adv-lambda', type=float, default=0.0,
        help='GRL max strength for CDAN (e.g., 0.1)')
    parser.add_argument('--entropy-cond', action='store_true',
        help='Entropy conditioning for CDAN (focus on confident tokens)')
    parser.add_argument('--freeze-encoder-n', type=int, default=0,
        help='Freeze first N encoder FeedForward blocks in stage 2')
    parser.add_argument('--stage2-lr', type=float, default=None,
        help='Optional learning-rate override in stage 2')

    # ---- OOS monitoring (new) ----
    parser.add_argument('--oos-sample', type=str, default=None,
        help="Path to ONE sample folder to evaluate out-of-sample validation loss on.")
    parser.add_argument('--oos-tmpdir', type=str, default=None,
        help="Optional temp batch dir for the OOS sample (reused if present).")
    parser.add_argument('--oos-batch-size', type=int, default=0,
        help="Optional explicit batch size for OOS evaluation.")
# ---- Back-compat: if user passes --dann-lambda, map to --adv-lambda
# (do this mapping in main() right after parsing)

    return parser.parse_args()


# ---------- Main ----------
def main():
    args = get_args()

    if getattr(args, 'dann_lambda', None) is not None and args.adv_lambda == 0.0:
        print('[compat] --dann-lambda detected; using as --adv-lambda')
        args.adv_lambda = args.dann_lambda


    # ----- Fast path: use existing batches without touching raw data -----
    batch_dir = os.path.join(args.prefix, 'batches')
    have_batches = os.path.isdir(batch_dir) and any(
        (f.startswith('batch_') and (f.endswith('_x.npy') or f.endswith('.pkl')))
        for f in os.listdir(batch_dir)
    )


    if have_batches and not args.force_rebuild_batches and not args.prepare_only:
        print(f"Using existing batches in {batch_dir} (fast path). Skipping raw sample scan.")
        # You can still pick a batch_size from the cached dataset:
        sample_dataset = CachedBatchDataset(batch_dir, max_cached_batches=2)
        batch_size = auto_batch_size(sample_dataset,  target_tokens=300000, hard_cap=128)

        del sample_dataset
        gc.collect()

        model_states = []
        for i in range(args.n_states):
            print(f"\nTraining model state {i+1}/{args.n_states}")
            model, _ = get_model_batched(
            batch_dir=batch_dir,
            prefix=f'{args.prefix}/states/{i:02d}/',
            batch_size=batch_size,
            epochs=(0 if args.two_stage else (args.epochs or 0)),
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
            monitor_metric=args.monitor_metric,
            patience=args.patience,
            resume_ckpt=args.resume_ckpt,
            # --- new ---
            entropy_cond=args.entropy_cond,
            two_stage=args.two_stage,
            epochs1=args.epochs1,
            epochs2=args.epochs2,
            freeze_encoder_n=args.freeze_encoder_n,
            stage2_lr=args.stage2_lr, 
            ###### oos monitor params ######
            oos_sample=args.oos_sample,
            oos_tmpdir=args.oos_tmpdir,
            oos_batch_size=args.oos_batch_size
        )

            model_states.append(model)
            if args.device == 'cuda':
                torch.cuda.empty_cache()

        print("\nStarting prediction...")
        predict_batched(
            model_states=model_states,
            batch_dir=batch_dir,
            prefix=args.prefix,
            device=args.device
        )
        print("Training and prediction completed!")
        return
    
    # ----- Build exclusion set -----
    exclude_set = set()
    if args.exclude_samples:
        exclude_set.update(s.strip() for s in args.exclude_samples.split(',') if s.strip())
    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    exclude_set.add(line)
    
    # ----- Build inclusion set (new) -----
    include_set = set()
    if args.include_samples:
        include_set.update(s.strip() for s in args.include_samples.split(',') if s.strip())

    if args.include_file and os.path.exists(args.include_file):
        with open(args.include_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    include_set.add(line)

    # ----- Collect candidate samples (adjust prefix if needed) -----

    all_subdirs = [d for d in os.listdir(args.prefix)
               if os.path.isdir(os.path.join(args.prefix, d))]

    # If include_set provided: start from those (that actually exist).
    # Else: default to all subdirs starting with 'P'.
    if include_set:
        base_names = sorted(list(set(all_subdirs) & include_set))
    else:
        base_names = [d for d in all_subdirs if d.startswith('P')]

    # Now apply exclusions
    kept_names = [d for d in base_names if d not in exclude_set]
    sample_dirs_all = sorted([os.path.join(args.prefix, d) + os.sep for d in kept_names])

    # Summaries
    included_requested = len(include_set)
    included_present = len(base_names) if include_set else None
    excluded_present = sorted(list(set(all_subdirs) & exclude_set))

    if include_set:
        missing_includes = sorted(list(include_set - set(all_subdirs)))
        print(f"Include filter requested {included_requested} names; present={included_present}, "
              f"missing={len(missing_includes)}")
        if missing_includes:
            print("Missing include names:", ", ".join(missing_includes))

    print(f"Found {len(sample_dirs_all)} candidate sample directories "
          f"(kept {len(kept_names)} of {len(all_subdirs)}; excluded present={len(excluded_present)})")
    if excluded_present:
        print("Excluded:", ", ".join(excluded_present))

    if not sample_dirs_all:
        print("ERROR: No sample directories after applying include/exclude filters.")
        return

    
    
    

    if not sample_dirs_all:
        print("ERROR: No sample directories after applying filter and exclusions.")
        return


    # ----- Align spots EARLY so downstream checks/readers see aligned files -----
    if args.align_spots:
        for d in sample_dirs_all:
            try:
                filter_and_save_shared_spots(d, overwrite=True, make_backup=True)
            except Exception as e:
                print(f"[align-spots] WARNING: {os.path.basename(os.path.normpath(d))}: {e}")


    # ----- Determine canonical radius from the first usable sample -----
    canonical_radius = None
    for d in sample_dirs_all:
        r = _read_radius(d)
        if r is not None:
            canonical_radius = r
            break
    if canonical_radius is None:
        print("ERROR: Could not find a valid radius.txt in any sample.")
        return
    print(f"Using canonical radius={canonical_radius}")

    # ----- Keep only complete/consistent samples; skip others with reasons -----
    sample_dirs = sample_dirs_all
    skipped = []
    # for d in sample_dirs_all:
    #     ok, reason = check_sample_complete(d, canonical_radius=canonical_radius)
    #     if ok:
    #         sample_dirs.append(d)
    #     else:
    #         skipped.append((os.path.basename(os.path.normpath(d)), reason))

    print(f"After completeness checks: {len(sample_dirs)} usable / {len(sample_dirs_all)} total")
    # if skipped:
    #     print("Skipped samples:")
    #     for name, why in skipped:
    #         print(f"  - {name}: {why}")

    if not sample_dirs:
        print("ERROR: No valid samples after completeness checks.")
        return

    # ----- Radius and batch dir -----
    radius = canonical_radius
    batch_dir = os.path.join(args.prefix, 'batches')

    # ----- Prepare batches if needed -----
    if not os.path.exists(batch_dir) or len(os.listdir(batch_dir)) == 0:
        print("Preparing and saving data batches...")
        batch_count = prepare_and_save_batches(
            sample_dirs, radius, batch_dir, args.samples_per_batch, domain_map_file=args.domain_map
        )
        if batch_count == 0:
            print("ERROR: No batches were created!")
            return
    else:
        print(f"Using existing batches in {batch_dir}")

    if args.prepare_only:
        print("Batch preparation completed. Use without --prepare-only to start training.")
        return

    # ----- Training -----
    print("\nStarting training with batched data...")
    sample_dataset = CachedBatchDataset(batch_dir, max_cached_batches=2)
    batch_size = auto_batch_size(sample_dataset, target_tokens=300000, hard_cap=128)

    del sample_dataset
    gc.collect()

    model_states = []
    for i in range(args.n_states):
        print(f"\nTraining model state {i+1}/{args.n_states}")
        model, _ = get_model_batched(
            batch_dir=batch_dir,
            prefix=f'{args.prefix}/states/{i:02d}/',
            batch_size=batch_size,
            epochs=(0 if args.two_stage else (args.epochs or 0)),
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
            monitor_metric=args.monitor_metric,
            patience=args.patience,
            resume_ckpt=args.resume_ckpt,
            # --- new ---
            entropy_cond=args.entropy_cond,
            two_stage=args.two_stage,
            epochs1=args.epochs1,
            epochs2=args.epochs2,
            freeze_encoder_n=args.freeze_encoder_n,
            stage2_lr=args.stage2_lr, 
            ######## oos monitor ########
            oos_sample=args.oos_sample,
            oos_tmpdir=args.oos_tmpdir,
            oos_batch_size=args.oos_batch_size
        )

        model_states.append(model)
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # # ----- Prediction -----
    # print("\nStarting prediction...")
    # predict_batched(
    #     model_states=model_states,
    #     batch_dir=batch_dir,
    #     prefix=args.prefix,
    #     device=args.device
    # )
    # print("Training and prediction completed!")



if __name__ == '__main__':
    main()
