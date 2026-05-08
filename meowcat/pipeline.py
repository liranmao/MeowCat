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
    """Return Visium sample names: CLI override > glob from data_root."""
    if override:
        return override
    pattern = os.path.join(cfg.project.data_root, cfg.visium.sample_pattern)
    matches = sorted(_glob.glob(pattern))
    return [os.path.basename(m) for m in matches if os.path.isdir(m)]


def _all_samples(cfg: MeowCatConfig, override: Optional[List[str]]) -> List[str]:
    """Discover both Visium + Xenium samples."""
    if override:
        return override
    vis = _samples(cfg, None)
    xen_pat = cfg.xenium.sample_pattern
    if xen_pat and xen_pat != cfg.visium.sample_pattern:
        pattern = os.path.join(cfg.project.data_root, xen_pat)
        xen = [os.path.basename(m) for m in sorted(_glob.glob(pattern)) if os.path.isdir(m)]
        vis_set = set(vis)
        vis.extend(s for s in xen if s not in vis_set)
    return vis


# ── Step 1 ─────────────────────────────────────────────────────────────────────
def cmd_rctd(cfg: MeowCatConfig) -> List[str]:
    p = cfg.project
    r = cfg.rctd
    cmd = [
        "Rscript", _pkg("Preprocess/RCTD_deconvolution.R"),
        "--base_dir", p.data_root,
        "--sample_pattern", cfg.visium.sample_pattern,
        "--reference_rds", r.reference_rds,
        "--cell_type_column", r.cell_type_column,
        "--max_cores", str(r.max_cores),
        "--doublet_mode", r.doublet_mode,
        "--min_umi", str(r.min_umi),
    ]
    if r.group_column:
        cmd += ["--group_column", r.group_column]
    if r.groups:
        cmd += ["--groups", ",".join(r.groups)]
    return cmd


# ── Step 2 ─────────────────────────────────────────────────────────────────────
def cmd_check_resolution(cfg: MeowCatConfig, samples: List[str]) -> List[str]:
    p = cfg.project
    pp = cfg.preprocess
    cmd = [
        "python", "-u", _pkg("Preprocess/ExtractFeatures/audit_resolution.py"),
        "--base_dir", p.data_root,
        "--samples", ",".join(samples),
        "--raw_flag", pp.raw_flag,
        "--target_mpp", str(pp.target_mpp),
    ]
    if pp.pixel_size_raw is not None:
        cmd += ["--pixel_size_raw", str(pp.pixel_size_raw)]
    return cmd


# ── Step 3 / 6a — per-sample preprocessing (train or predict images) ──────────
def cmds_preprocess_sample(cfg: MeowCatConfig, sample: str) -> List[List[str]]:
    """Returns an ordered list of commands to preprocess one sample."""
    data_root = cfg.project.data_root
    # Trailing separator so read_dir + raw_flag produces a valid path prefix
    # (scripts use prefix = read_dir + raw_flag, e.g. "/path/sample/" + "he_raw")
    feature_dir = os.path.join(data_root, sample, "")
    pp = cfg.preprocess

    get_pixel_size = [
        "python", "-u", _pkg("Preprocess/ExtractFeatures/get_pixel_size.py"),
        "--read_dir", feature_dir,
        "--save_dir", feature_dir,
        "--sample", sample,
        "--raw_flag", pp.raw_flag,
    ]
    if pp.pixel_size_raw is not None:
        get_pixel_size += ["--pixel_size_raw", str(pp.pixel_size_raw)]

    run_preprocess = [
        "python", "-u", _pkg("Preprocess/ExtractFeatures/RunPreprocess.py"),
        "--read_dir", feature_dir,
        "--save_dir", feature_dir,
        "--sample", sample,
        "--raw_flag", pp.raw_flag,
        "--pad", str(pp.pad),
        # scale_value is derived at runtime from pixel-size-raw.txt; the CLI
        # wrapper reads it after get_pixel_size runs.
        "--scale_value", "{scale}",  # placeholder resolved in cli.py
    ]

    # RunHistoSweep saves masks to {read_dir}/mask/ — UNI_extract_features.py
    # hardcodes the same {read_path}/mask/ lookup, so this must always be "mask".
    run_histosweep = [
        "python", "-u", _pkg("Preprocess/ExtractFeatures/RunHistoSweep.py"),
        "--read_dir", feature_dir,
        "--save_dir", "mask",
    ]

    extract_features = [
        "python", "-u", _pkg("Preprocess/ExtractFeatures/UNI_extract_features.py"),
        "--read_path", feature_dir,
        "--save_dir", feature_dir,
        "--weight_dir", pp.uni_weights,
        "--sample", sample,
    ]

    fuse_features = [
        "python", "-u", _pkg("Preprocess/ExtractFeatures/UNI_fuse_features.py"),
        "--read_global_path", feature_dir,
        "--read_local_path", feature_dir,
        "--save_dir", feature_dir,
        "--sample", sample,
        "--mode", pp.fusion_mode,
    ]

    return [get_pixel_size, run_preprocess, run_histosweep,
            extract_features, fuse_features]


# ── Step 1.5 — Visium-specific input preparation (per sample) ─────────────────
def cmds_prepare_visium_sample(cfg: MeowCatConfig, sample: str) -> List[List[str]]:
    """
    Prepare all Visium-specific files for one sample.

    Returns an ordered list of two commands:

    1. ``prepare_visium_inputs.py`` — RCTD + spatial → anno-names.txt, anno_matrix.tsv, locs.tsv, radius.txt, pixel-size.txt
    2. ``prepare_inference_new_sample.py`` — single_super_emb.h5ad → embeddings-hist.pickle or .npy

    Non-Visium samples are skipped gracefully by prepare_visium_inputs.py.
    Requires pixel-size-raw.txt to already exist (written by get_pixel_size.py).
    """
    data_root = cfg.project.data_root
    feature_dir = os.path.join(data_root, sample)

    prepare_visium = [
        "python", "-u", _pkg("Preprocess/prepare_visium_inputs.py"),
        "--sample_dir", feature_dir,
        "--target_mpp", str(cfg.preprocess.target_mpp),
    ]

    prepare_embeddings = [
        "python", "-u", _pkg("Preprocess/prepare_inference_new_sample.py"),
        data_root,
        sample,
    ]

    return [prepare_visium, prepare_embeddings]


# ── Step 1.6 — Visium QC visualization ────────────────────────────────────────
def cmd_visualize_visium(
    cfg: MeowCatConfig,
    samples: Optional[List[str]] = None,
) -> List[str]:
    """
    Overlay Visium spots (locs.tsv, radius.txt, anno_matrix.tsv) on the
    processed H&E image to verify pixel-size scaling.
    Outputs go to <out_root>/<sample>/visium_viz/.
    """
    cmd = [
        "python", "-u", _pkg("Preprocess/visualize_visium_prep.py"),
        "--data_root",      cfg.project.data_root,
        "--out_root",       cfg.project.out_root,
        "--sample_pattern", cfg.visium.sample_pattern,
    ]
    if samples:
        cmd += ["--samples", ",".join(samples)]
    return cmd


# ── Step 4 — Visium batch preparation ─────────────────────────────────────────
def cmd_prepare_visium_batches(cfg: MeowCatConfig, config_path: str) -> List[str]:
    return [
        "python", "-u", _pkg("Preprocess/batched_data_preparing.py"),
        "--config", config_path,
        "--mode", "visium",
    ]


# ── Step 4x — Xenium batch preparation ───────────────────────────────────────
def cmd_prepare_xenium_batches(cfg: MeowCatConfig, config_path: str) -> List[str]:
    return [
        "python", "-u", _pkg("Preprocess/batched_data_preparing.py"),
        "--config", config_path,
        "--mode", "xenium",
    ]


# ── Slim Xenium cellbins ─────────────────────────────────────────────────────
def cmd_slim_xenium(cfg: MeowCatConfig, config_path: str) -> List[str]:
    return [
        "python", "-u", _pkg("Preprocess/batched_data_preparing.py"),
        "--config", config_path,
        "--mode", "slim-xenium",
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
    elif t.two_stage:
        # Non-sequential mode: pass epochs2 for Stage 2 training
        epochs2 = max(t.visium_epochs, t.xenium_epochs)
        if epochs2 > 0:
            cmd += ["--epochs2", str(epochs2)]
    if t.oos_sample:
        cmd += ["--oos-sample", t.oos_sample]
    if t.oos_tmpdir:
        cmd += ["--oos-tmpdir", t.oos_tmpdir]
    return cmd


# ── Step 6b — predict one sample ───────────────────────────────────────────────
def cmd_predict_sample(cfg: MeowCatConfig, sample: str) -> List[str]:
    p = cfg.predict
    return [
        "python", "-u",
        _pkg("Train_predict/predict_cdan_multireso.py"),
        cfg.batches.out_dir,
        str(p.n_states),
        sample,
        "--data-root", cfg.project.data_root,
        "--device", p.device,
        "--tokens-per-chunk", str(p.tokens_per_chunk),
        "--chunks-per-batch", str(p.chunks_per_batch),
        "--out-pkl-name", p.out_pkl_name,
    ]


# ── Step 6b — visualize one sample ────────────────────────────────────────────
def cmd_visualize_sample(cfg: MeowCatConfig, sample: str) -> List[str]:
    v = cfg.visualize
    p = cfg.project

    # Resolve cmap JSON: explicit config > bundled default
    cmap_json = v.cmap_json
    if cmap_json is None:
        cmap_json = os.path.join(_HERE, "..", "config", "visualization_cmap.json")

    cmd = [
        "python", "-u",
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
        "--cmap-json", cmap_json,
    ]
    if v.save_highlights:
        cmd.append("--save-highlights")
    return cmd


# ── Step 7 ─────────────────────────────────────────────────────────────────────
def cmd_slide(cfg: MeowCatConfig) -> List[str]:
    s = cfg.slide
    pptx = s.pptx if os.path.isabs(s.pptx) else os.path.join(cfg.project.out_root, s.pptx)
    # Discover samples so slide script gets explicit names
    sample_names = _all_samples(cfg, None)
    cmd = [
        "python", "-u", _pkg("Train_predict/visualize_slide_wrap.py"),
        "--out-root", cfg.project.out_root,
        "--pptx", pptx,
        "--intensity-cols", str(s.intensity_cols),
        "--intensity-rows", str(s.intensity_rows),
        "--highlight-cols", str(s.highlight_cols),
        "--highlight-rows", str(s.highlight_rows),
    ]
    if sample_names:
        cmd += ["--samples", ",".join(sample_names)]
    return cmd
