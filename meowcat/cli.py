"""
meowcat/cli.py
Unified CLI entry point.  All subcommands read from a single config YAML and
call existing scripts as subprocesses — no core logic lives here.

Usage
-----
    meowcat <step> --config config/my_run.yaml [--samples S1,S2] [--dry-run]

Steps (in pipeline order):
    infer                   Predict on new H&E images using a trained model
    rctd                    Step 1   — RCTD deconvolution (R)
    prepare-visium          Step 1.5 — Visium metadata prep (anno_matrix, locs, radius files)
    visualize-visium        Step 1.6 — QC: overlay Visium spots on processed H&E
    check-resolution        Step 2   — audit image resolutions
    preprocess              Step 3/6a — image preprocess for training OR prediction samples
    prepare-visium-batches  Step 4   — build Visium training batch files
    prepare-xenium-batches  Step 4x  — build Xenium training batch files
    train                   Step 5   — train MeowCat models
    predict                 Step 6b  — run full-grid prediction
    visualize               Step 6b  — visualize prediction outputs
    slide                   Step 7   — generate PowerPoint summary
    run-all                 Steps 1-7 in sequence
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional

from .config import load_config, MeowCatConfig
from . import pipeline as _pl

# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: List[str], dry_run: bool) -> None:
    """Print and optionally execute a command."""
    print("$ " + " ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)


def _parse_samples(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


# ── subcommand handlers ───────────────────────────────────────────────────────

def cmd_rctd(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    _run(_pl.cmd_rctd(cfg), args.dry_run)


def cmd_prepare_visium(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    """
    Step 1.5 — Visium metadata + embeddings preparation (per sample).

    For each sample, runs:
      1. prepare_visium_inputs.py — writes anno-names.txt, anno_matrix.tsv,
         locs.tsv, radius-raw.txt, radius.txt, pixel-size.txt
      2. prepare_inference_new_sample.py — converts single_super_emb.h5ad
         to embeddings-hist.pickle

    Requires pixel-size-raw.txt to already exist in the sample directory
    (written by get_pixel_size.py, the first sub-step of 'meowcat preprocess',
    or created manually with the raw MPP value).

    Non-Visium samples are skipped automatically by step 1.
    """
    samples = _parse_samples(args.samples) or _pl._all_samples(cfg, None)
    if not samples:
        print("[meowcat] No samples found. Check project.data_root and visium/xenium sample_pattern.")
        sys.exit(1)

    for sample in samples:
        print(f"\n[meowcat] Preparing Visium inputs: {sample}")
        for cmd in _pl.cmds_prepare_visium_sample(cfg, sample):
            _run(cmd, args.dry_run)


def cmd_visualize_visium(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    """
    Step 1.6 — Visium QC visualization.
    Overlays spots on the processed H&E image to confirm pixel-size scaling.
    Outputs: <out_root>/<sample>/visium_viz/celltype_*.png + argmax_spots.png
    """
    samples = _parse_samples(args.samples)
    _run(_pl.cmd_visualize_visium(cfg, samples), args.dry_run)


def cmd_check_resolution(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    samples = _parse_samples(getattr(args, "samples", None)) or _pl._all_samples(cfg, None)
    if not samples:
        print("[meowcat] No samples found. Check visium/xenium sample_pattern and project.data_root.")
        sys.exit(1)
    _run(_pl.cmd_check_resolution(cfg, samples), args.dry_run)


_PREPROCESS_SUBSTEPS = [
    "get_pixel_size",       # 1
    "run_preprocess",       # 2
    "run_histosweep",       # 3
    "extract_features",     # 4
    "fuse_features",        # 5
]


def cmd_preprocess(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    samples = _parse_samples(args.samples) or _pl._all_samples(cfg, None)
    if not samples:
        print("[meowcat] No samples found. Check project.data_root and visium/xenium sample_pattern.")
        sys.exit(1)

    start_from = getattr(args, "start_from", 1)

    for sample in samples:
        print(f"\n[meowcat] Preprocessing: {sample}")
        cmds = _pl.cmds_preprocess_sample(cfg, sample)
        feature_dir = os.path.join(cfg.project.data_root, sample)

        for i, cmd in enumerate(cmds):
            step_num = i + 1
            step_name = _PREPROCESS_SUBSTEPS[i] if i < len(_PREPROCESS_SUBSTEPS) else f"step_{step_num}"

            if step_num < start_from:
                print(f"  [skip] substep {step_num}/{len(cmds)} ({step_name}) — skipped by --start-from {start_from}")
                continue

            # Step 2 (RunPreprocess) needs the actual pixel size computed by step 1
            if "{scale}" in cmd:
                pixel_file = os.path.join(feature_dir, "pixel-size-raw.txt")
                if not args.dry_run:
                    if not os.path.exists(pixel_file):
                        print(f"  [skip] pixel-size-raw.txt not found for {sample}; "
                              f"run get_pixel_size first.")
                        break
                    with open(pixel_file) as f:
                        pixel_size_raw = float(f.read().strip())
                    scale = str(pixel_size_raw / cfg.preprocess.target_mpp)
                else:
                    scale = "<pixel_size_raw / target_mpp>"
                cmd = [c.replace("{scale}", scale) for c in cmd]

            print(f"  [substep {step_num}/{len(cmds)}] {step_name}")
            _run(cmd, args.dry_run)


def cmd_prepare_visium_batches(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    _run(_pl.cmd_prepare_visium_batches(cfg, args.config), args.dry_run)


def cmd_prepare_xenium_batches(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    """
    Step 4x — Xenium batch preparation.

    Runs Xenium preliminary processing (cellbin + annotation → h5ad) and
    batch creation (h5ad + embeddings → batch_xen_*_{x,y,d}.npy).
    Reads paths and parameters from the 'xenium' config section.
    """
    _run(_pl.cmd_prepare_xenium_batches(cfg, args.config), args.dry_run)


def cmd_train(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    _run(_pl.cmd_train(cfg), args.dry_run)


def cmd_predict(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    samples = _parse_samples(args.samples) or _pl._all_samples(cfg, None)
    for sample in samples:
        print(f"\n[meowcat] Predicting: {sample}")
        _run(_pl.cmd_predict_sample(cfg, sample), args.dry_run)


def cmd_visualize(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    samples = _parse_samples(args.samples) or _pl._all_samples(cfg, None)
    for sample in samples:
        print(f"\n[meowcat] Visualizing: {sample}")
        _run(_pl.cmd_visualize_sample(cfg, sample), args.dry_run)


def cmd_slide(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    _run(_pl.cmd_slide(cfg), args.dry_run)


def cmd_infer(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    """
    Run cell-type prediction on new H&E images using a trained model.

    Chains: preprocess -> prepare-visium (embeddings) -> predict -> visualize
    per sample, then generates a summary slide.
    """
    inf = cfg.inference
    if not inf.model_dir or not inf.anno_names:
        print("ERROR: inference.model_dir and inference.anno_names are required")
        sys.exit(1)
    if not os.path.isfile(inf.anno_names):
        print(f"ERROR: anno-names file not found: {inf.anno_names}")
        sys.exit(1)

    # Override batches.out_dir so predict/slide find the trained model checkpoints
    cfg.batches.out_dir = inf.model_dir

    samples = _parse_samples(args.samples) or _pl._samples(cfg, None)
    if not samples:
        print("[meowcat] No samples found. Check project.data_root and visium/xenium sample_pattern.")
        sys.exit(1)

    start_from = getattr(args, "start_from", 1)

    for sample in samples:
        sample_dir = os.path.join(cfg.project.data_root, sample)

        # Copy anno-names.txt into sample dir if missing
        anno_dst = os.path.join(sample_dir, "anno-names.txt")
        if not os.path.exists(anno_dst):
            if not args.dry_run:
                os.makedirs(sample_dir, exist_ok=True)
                shutil.copy2(inf.anno_names, anno_dst)
            print(f"[meowcat] Copied anno-names.txt -> {anno_dst}")

        # 1. Preprocess
        if start_from <= 5:
            print(f"\n[meowcat] Preprocessing: {sample}")
            cmds = _pl.cmds_preprocess_sample(cfg, sample)
            feature_dir = os.path.join(cfg.project.data_root, sample)
            for i, cmd in enumerate(cmds):
                step_num = i + 1
                step_name = _PREPROCESS_SUBSTEPS[i] if i < len(_PREPROCESS_SUBSTEPS) else f"step_{step_num}"
                if step_num < start_from:
                    print(f"  [skip] substep {step_num}/{len(cmds)} ({step_name}) — skipped by --start-from {start_from}")
                    continue
                if "{scale}" in cmd:
                    pixel_file = os.path.join(feature_dir, "pixel-size-raw.txt")
                    if not args.dry_run:
                        if not os.path.exists(pixel_file):
                            print(f"  [skip] pixel-size-raw.txt not found for {sample}; "
                                  f"run get_pixel_size first.")
                            break
                        with open(pixel_file) as f:
                            pixel_size_raw = float(f.read().strip())
                        scale = str(pixel_size_raw / cfg.preprocess.target_mpp)
                    else:
                        scale = "<pixel_size_raw / target_mpp>"
                    cmd = [c.replace("{scale}", scale) for c in cmd]
                print(f"  [substep {step_num}/{len(cmds)}] {step_name}")
                _run(cmd, args.dry_run)

        # 2. Prepare embeddings (prepare-visium: step 1 skips for non-Visium, step 2 creates embeddings-hist.pickle)
        print(f"\n[meowcat] Preparing embeddings: {sample}")
        for cmd in _pl.cmds_prepare_visium_sample(cfg, sample):
            _run(cmd, args.dry_run)

        # 3. Predict
        print(f"\n[meowcat] Predicting: {sample}")
        _run(_pl.cmd_predict_sample(cfg, sample), args.dry_run)

        # 4. Visualize
        print(f"\n[meowcat] Visualizing: {sample}")
        _run(_pl.cmd_visualize_sample(cfg, sample), args.dry_run)

    # 5. Slide
    print("\n[meowcat] Generating summary slide")
    _run(_pl.cmd_slide(cfg), args.dry_run)


def cmd_run_all(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    # run-all doesn't register --samples or --start-from, so set defaults
    if not hasattr(args, "samples"):
        args.samples = None
    if not hasattr(args, "start_from"):
        args.start_from = 1
    cmd_rctd(cfg, args)
    cmd_check_resolution(cfg, args)
    cmd_preprocess(cfg, args)
    cmd_prepare_visium(cfg, args)
    cmd_prepare_visium_batches(cfg, args)
    cmd_prepare_xenium_batches(cfg, args)
    cmd_train(cfg, args)
    cmd_predict(cfg, args)
    cmd_visualize(cfg, args)
    cmd_slide(cfg, args)


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    # Shared arguments available on every subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default=None, metavar="YAML",
        help="Path to config YAML (default: config/default.yaml)",
    )
    common.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )

    root = argparse.ArgumentParser(
        prog="meowcat",
        description="MeowCat: H&E cell-type annotation pipeline.",
    )

    sub = root.add_subparsers(dest="step", required=True)

    # steps with no extra args
    for name, help_text in [
        ("rctd",                    "Step 1: RCTD deconvolution (R)"),
        ("prepare-visium-batches",  "Step 4: build Visium training batch files"),
        ("prepare-xenium-batches",  "Step 4x: build Xenium training batch files"),
        ("train",                   "Step 5: train MeowCat models"),
        ("slide",                   "Step 7: generate PowerPoint summary"),
        ("run-all",                 "Run all steps in order (1→7)"),
    ]:
        sub.add_parser(name, help=help_text, parents=[common])

    # steps that operate per-sample (or accept --samples filter)
    for name, help_text in [
        ("check-resolution",  "Step 2: audit H&E image resolutions"),
        ("prepare-visium",    "Step 1.5: Visium metadata prep (anno_matrix, locs, radius files)"),
        ("visualize-visium",  "Step 1.6: QC — overlay Visium spots on processed H&E image"),
        ("predict",           "Step 6b: run full-grid cell-type prediction"),
        ("visualize",         "Step 6b: visualize prediction outputs"),
    ]:
        p = sub.add_parser(name, help=help_text, parents=[common])
        p.add_argument(
            "--samples", default=None, metavar="S1,S2",
            help="Comma-separated sample names (default: all matching visium/xenium sample_pattern)",
        )

    # infer has --samples AND --start-from
    p = sub.add_parser("infer",
                        help="Run prediction on new H&E images using a trained model",
                        parents=[common])
    p.add_argument(
        "--samples", default=None, metavar="S1,S2",
        help="Comma-separated sample names (default: all matching visium/xenium sample_pattern)",
    )
    p.add_argument(
        "--start-from", type=int, default=1, metavar="N",
        help=(
            "Resume preprocessing from substep N (1-5, default: 1). "
            "Substeps: 1=get_pixel_size, 2=run_preprocess, 3=run_histosweep, "
            "4=extract_features, 5=fuse_features. "
            "Use 6 to skip preprocessing entirely."
        ),
    )

    # preprocess has --samples AND --start-from
    p = sub.add_parser("preprocess",
                        help="Step 3/6a: preprocess H&E images (training or prediction samples)",
                        parents=[common])
    p.add_argument(
        "--samples", default=None, metavar="S1,S2",
        help="Comma-separated sample names (default: all matching visium/xenium sample_pattern)",
    )
    p.add_argument(
        "--start-from", type=int, default=1, metavar="N",
        help=(
            "Resume from substep N (1-5, default: 1). "
            "Substeps: 1=get_pixel_size, 2=run_preprocess, 3=run_histosweep, "
            "4=extract_features, 5=fuse_features"
        ),
    )

    return root


# ── entry point ───────────────────────────────────────────────────────────────

_HANDLERS = {
    "infer":                    cmd_infer,
    "rctd":                     cmd_rctd,
    "prepare-visium":           cmd_prepare_visium,
    "visualize-visium":         cmd_visualize_visium,
    "check-resolution":         cmd_check_resolution,
    "preprocess":               cmd_preprocess,
    "prepare-visium-batches":   cmd_prepare_visium_batches,
    "prepare-xenium-batches":   cmd_prepare_xenium_batches,
    "train":                    cmd_train,
    "predict":                  cmd_predict,
    "visualize":                cmd_visualize,
    "slide":                    cmd_slide,
    "run-all":                  cmd_run_all,
}


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _HANDLERS[args.step](cfg, args)


if __name__ == "__main__":
    main()
