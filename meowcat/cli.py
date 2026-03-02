"""
meowcat/cli.py
Unified CLI entry point.  All subcommands read from a single config YAML and
call existing scripts as subprocesses — no core logic lives here.

Usage
-----
    meowcat <step> --config config/my_run.yaml [--samples S1,S2] [--dry-run]

Steps (in pipeline order):
    rctd                    Step 1   — RCTD deconvolution (R)
    prepare-visium          Step 1.5 — Visium metadata prep (anno_matrix, locs, radius files)
    visualize-visium        Step 1.6 — QC: overlay Visium spots on processed H&E
    check-resolution        Step 2   — audit image resolutions
    preprocess              Step 3/6a — image preprocess for training OR prediction samples
    prepare-batches         Step 4   — build Visium training batch files
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
    Step 1.5 — Visium metadata preparation (per sample).

    For each sample, writes: anno-names.txt, anno_matrix.tsv, locs.tsv,
    radius-raw.txt, radius.txt, pixel-size.txt.

    Requires pixel-size-raw.txt to already exist in the sample directory
    (written by get_pixel_size.py, the first sub-step of 'meowcat preprocess',
    or created manually with the raw MPP value).

    Non-Visium samples are skipped automatically.
    """
    samples = _parse_samples(args.samples) or _pl._samples(cfg, None)
    if not samples:
        print("[meowcat] No samples found. Check project.data_root and project.sample_pattern.")
        sys.exit(1)

    for sample in samples:
        print(f"\n[meowcat] Preparing Visium inputs: {sample}")
        _run(_pl.cmd_prepare_visium_sample(cfg, sample), args.dry_run)


def cmd_visualize_visium(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    """
    Step 1.6 — Visium QC visualization.
    Overlays spots on the processed H&E image to confirm pixel-size scaling.
    Outputs: <out_root>/<sample>/visium_viz/celltype_*.png + argmax_spots.png
    """
    samples = _parse_samples(args.samples)
    _run(_pl.cmd_visualize_visium(cfg, samples), args.dry_run)


def cmd_check_resolution(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    _run(_pl.cmd_check_resolution(cfg), args.dry_run)


def cmd_preprocess(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    samples = _parse_samples(args.samples) or _pl._samples(cfg, None)
    if not samples:
        print("[meowcat] No samples found. Check project.data_root and project.sample_pattern.")
        sys.exit(1)

    for sample in samples:
        print(f"\n[meowcat] Preprocessing: {sample}")
        cmds = _pl.cmds_preprocess_sample(cfg, sample)
        feature_dir = os.path.join(cfg.project.data_root, sample)

        for i, cmd in enumerate(cmds):
            # Step 1 (RunPreprocess) needs the actual pixel size computed by step 0
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

            _run(cmd, args.dry_run)


def cmd_prepare_batches(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    _run(_pl.cmd_prepare_batches(cfg, args.config), args.dry_run)


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
    samples = _parse_samples(args.samples) or _pl._samples(cfg, None)
    for sample in samples:
        print(f"\n[meowcat] Predicting: {sample}")
        _run(_pl.cmd_predict_sample(cfg, sample), args.dry_run)


def cmd_visualize(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    samples = _parse_samples(args.samples) or _pl._samples(cfg, None)
    for sample in samples:
        print(f"\n[meowcat] Visualizing: {sample}")
        _run(_pl.cmd_visualize_sample(cfg, sample), args.dry_run)


def cmd_slide(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    _run(_pl.cmd_slide(cfg), args.dry_run)


def cmd_run_all(cfg: MeowCatConfig, args: argparse.Namespace) -> None:
    cmd_rctd(cfg, args)
    cmd_check_resolution(cfg, args)
    # preprocess already runs prepare_visium as its last sub-step for Visium samples
    cmd_preprocess(cfg, args)
    cmd_prepare_batches(cfg, args)
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
        ("check-resolution",        "Step 2: audit H&E image resolutions"),
        ("prepare-batches",         "Step 4: build Visium training batch files"),
        ("prepare-xenium-batches",  "Step 4x: build Xenium training batch files"),
        ("train",                   "Step 5: train MeowCat models"),
        ("slide",                   "Step 7: generate PowerPoint summary"),
        ("run-all",                 "Run all steps in order (1→7)"),
    ]:
        sub.add_parser(name, help=help_text, parents=[common])

    # steps that operate per-sample (or accept --samples filter)
    for name, help_text in [
        ("prepare-visium",    "Step 1.5: Visium metadata prep (anno_matrix, locs, radius files)"),
        ("visualize-visium",  "Step 1.6: QC — overlay Visium spots on processed H&E image"),
        ("preprocess",        "Step 3/6a: preprocess H&E images (training or prediction samples)"),
        ("predict",           "Step 6b: run full-grid cell-type prediction"),
        ("visualize",         "Step 6b: visualize prediction outputs"),
    ]:
        p = sub.add_parser(name, help=help_text, parents=[common])
        p.add_argument(
            "--samples", default=None, metavar="S1,S2",
            help="Comma-separated sample names (default: all matching project.sample_pattern)",
        )

    return root


# ── entry point ───────────────────────────────────────────────────────────────

_HANDLERS = {
    "rctd":                     cmd_rctd,
    "prepare-visium":           cmd_prepare_visium,
    "visualize-visium":         cmd_visualize_visium,
    "check-resolution":         cmd_check_resolution,
    "preprocess":               cmd_preprocess,
    "prepare-batches":          cmd_prepare_batches,
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
