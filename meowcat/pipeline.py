"""
meowcat/pipeline.py
Builds subprocess argument lists for each pipeline step from a MeowCatConfig.
The actual subprocess.run() call is in cli.py so dry-run works cleanly.
"""

from __future__ import annotations
import os
import glob as _glob
from typing import List, Optional

from .config import MeowCatConfig

# ── locate this package directory (meowcat/) ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def _pkg(relpath: str) -> str:
    """Resolve a path relative to the meowcat/ package directory."""
    return os.path.join(_HERE, relpath)


def _samples(cfg: MeowCatConfig, override: Optional[List[str]]) -> List[str]:
    """Return list of sample names: CLI override > glob from data_root."""
    if override:
        return override
    pattern = os.path.join(cfg.project.data_root, cfg.project.sample_pattern)
    matches = sorted(_glob.glob(pattern))
    return [os.path.basename(m) for m in matches if os.path.isdir(m)]


# ── Step 1 ─────────────────────────────────────────────────────────────────────
def cmd_rctd(cfg: MeowCatConfig) -> List[str]:
    return ["Rscript", _pkg("Preprocess/RCTD_deconvolution.R"), "--no-save"]


# ── Step 2 ─────────────────────────────────────────────────────────────────────
def cmd_check_resolution(cfg: MeowCatConfig) -> List[str]:
    p = cfg.project
    pp = cfg.preprocess
    return [
        "python", _pkg("Preprocess/ExtractFeatures/audit_resolution.py"),
        "--base_dir", p.data_root,
        "--pattern", p.sample_pattern,
        "--raw_flag", pp.raw_flag,
        "--target_mpp", str(pp.target_mpp),
    ]


# ── Step 3 / 6a — per-sample preprocessing (train or predict images) ──────────
def cmds_preprocess_sample(cfg: MeowCatConfig, sample: str) -> List[List[str]]:
    """Returns an ordered list of commands to preprocess one sample."""
    data_root = cfg.project.data_root
    feature_dir = os.path.join(data_root, sample)
    pp = cfg.preprocess

    get_pixel_size = [
        "python", _pkg("Preprocess/ExtractFeatures/get_pixel_size.py"),
        "--read_dir", feature_dir,
        "--save_dir", feature_dir,
        "--sample", sample,
        "--raw_flag", pp.raw_flag,
    ]

    run_preprocess = [
        "python", _pkg("Preprocess/ExtractFeatures/RunPreprocess.py"),
        "--read_dir", feature_dir,
        "--save_dir", feature_dir,
        "--sample", sample,
        "--raw_flag", pp.raw_flag,
        "--pad", str(pp.pad),
        # scale_value is derived at runtime from pixel-size-raw.txt; the CLI
        # wrapper reads it after get_pixel_size runs.
        "--scale_value", "{scale}",  # placeholder resolved in cli.py
    ]

    run_histosweep = [
        "python", _pkg("Preprocess/ExtractFeatures/RunHistoSweep.py"),
        "--read_dir", feature_dir,
        "--save_dir", pp.histosweep_mask_dir,
    ]

    extract_features = [
        "python", _pkg("Preprocess/ExtractFeatures/UNI_extract_features.py"),
        "--read_path", feature_dir,
        "--save_dir", feature_dir,
        "--weight_dir", pp.uni_weights,
        "--sample", sample,
    ]

    fuse_features = [
        "python", _pkg("Preprocess/ExtractFeatures/UNI_fuse_features.py"),
        "--read_global_path", feature_dir,
        "--read_local_path", feature_dir,
        "--save_dir", feature_dir,
        "--sample", sample,
        "--mode", pp.fusion_mode,
    ]

    prepare_inference = [
        "python", _pkg("Preprocess/prepare_inference_new_sample.py"),
        data_root, sample,
    ]

    return [get_pixel_size, run_preprocess, run_histosweep,
            extract_features, fuse_features, prepare_inference]


# ── Step 4 ─────────────────────────────────────────────────────────────────────
def cmd_prepare_batches(cfg: MeowCatConfig, config_path: str) -> List[str]:
    return [
        "python", _pkg("Preprocess/batched_data_preparing.py"),
        "--config", config_path,
    ]


# ── Step 5 ─────────────────────────────────────────────────────────────────────
def cmd_train(cfg: MeowCatConfig) -> List[str]:
    t = cfg.train
    cmd = [
        "python", "-u",
        _pkg("Train_predict/train_by_batch_cdan5_trainc_final2.py"),
        cfg.batches.out_dir,
        "--n-states", str(t.n_states),
        "--adv-lambda", str(t.adv_lambda),
        "--freeze-encoder-n", str(t.freeze_encoder_n),
        "--recon-weight", str(t.recon_weight),
        "--recon-mask-ratio", str(t.recon_mask_ratio),
        "--save-every-n-epochs", str(t.save_every_n_epochs),
        "--xenium-weight", str(t.xenium_weight),
        "--monitor-metric", t.monitor_metric,
        "--device", t.device,
    ]
    if t.two_stage:
        cmd += ["--two-stage", "--epochs1", str(t.epochs1)]
    if t.sequential_training:
        cmd += [
            "--sequential-training",
            "--visium-epochs", str(t.visium_epochs),
            "--xenium-epochs", str(t.xenium_epochs),
        ]
    if t.oos_sample:
        cmd += ["--oos-sample", t.oos_sample]
    if t.oos_tmpdir:
        cmd += ["--oos-tmpdir", t.oos_tmpdir]
    return cmd


# ── Step 6b — predict one sample ───────────────────────────────────────────────
def cmd_predict_sample(cfg: MeowCatConfig, sample: str) -> List[str]:
    p = cfg.predict
    return [
        "python",
        _pkg("Train_predict/predict_cdan_multireso.py"),
        cfg.project.data_root,
        str(p.n_states),
        sample,
        "--device", p.device,
        "--tokens-per-chunk", str(p.tokens_per_chunk),
        "--chunks-per-batch", str(p.chunks_per_batch),
        "--out-pkl-name", p.out_pkl_name,
    ]


# ── Step 6b — visualize one sample ────────────────────────────────────────────
def cmd_visualize_sample(cfg: MeowCatConfig, sample: str) -> List[str]:
    v = cfg.visualize
    p = cfg.project
    cmd = [
        "python",
        _pkg("Train_predict/visualize_prediction_results.py"),
        "--data-root", p.data_root,
        "--data-root-ori", p.data_root,
        "--sample", sample,
        "--pkl-name", cfg.predict.out_pkl_name,
        "--out-root", p.out_root,
        "--n-clusters", str(v.n_clusters),
        "--pca-comp", str(v.pca_comp),
        "--random-seed", str(v.random_seed),
        "--p-lo", str(v.p_lo),
        "--p-hi", str(v.p_hi),
    ]
    if v.save_highlights:
        cmd.append("--save-highlights")
    return cmd


# ── Step 7 ─────────────────────────────────────────────────────────────────────
def cmd_slide(cfg: MeowCatConfig) -> List[str]:
    s = cfg.slide
    pptx = s.pptx if os.path.isabs(s.pptx) else os.path.join(cfg.project.out_root, s.pptx)
    return [
        "python", _pkg("Train_predict/visualize_slide_wrap.py"),
        "--out-root", cfg.project.out_root,
        "--pptx", pptx,
        "--intensity-cols", str(s.intensity_cols),
        "--intensity-rows", str(s.intensity_rows),
        "--highlight-cols", str(s.highlight_cols),
        "--highlight-rows", str(s.highlight_rows),
    ]
