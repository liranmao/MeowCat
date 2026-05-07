#!/usr/bin/env bash
# =============================================================================
# Example 4a: Train on multiple Visium samples (no Xenium)
#             Using pre-prepared batches and embeddings from 7_new_sc_data
#
# Prerequisites:
#   1. Run setup_data.sh first to symlink samples and convert batches
#   2. Activate: conda activate he_anno
#
# Training paradigm:
#   Phase 0 (Recon):  reconstruction pretraining              (15 epochs)
#   Phase 1 (Visium): MSE on RCTD soft proportions + CDAN     (100 epochs)
#
# CDAN aligns patient-level domains (each WSI = one domain) so the encoder
# learns patient-invariant features.
#
# Input layout (created by setup_data.sh):
#   /project/KidneyHE/01_meowcat_test/04_multi_visium/input/
#     P001/ -> (symlink) /project/KidneyHE/data_lung/7_new_sc_data/P001/
#     P002/ -> ...
#     Each sample contains: embeddings-hist.pickle or .npy, anno-names.txt, mask/
#
# Output layout:
#   /project/KidneyHE/01_meowcat_test/04_multi_visium/output/
#     batches/
#       batch_vis_000_x/y/d.npy      <- domain 0
#       batch_vis_001_x/y/d.npy      <- domain 1
#       ...
#       states/00/model.ckpt
#       states/01/model.ckpt
#     results_ex04_multi_visium.pptx
# =============================================================================
set -euo pipefail
export PYTHONWARNINGS="ignore"

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 4a: Multi-Visium training"
echo " Config: $CFG"
echo " (Steps 1-4 skipped — using pre-prepared data)"
echo "============================================"

# Steps 1-4 are SKIPPED: data already prepared via setup_data.sh
# (RCTD, preprocess, prepare-visium, prepare-visium-batches)

# ---------------------------------------------------------------------------
# Step 5: Train — Recon + Visium MSE + CDAN
# Activate: conda activate he_anno
# Phase 0 (15 ep):   reconstruction pretraining
# Phase 1 (100 ep):  Visium MSE + CDAN domain adversarial (adv_lambda=0.005)
# ---------------------------------------------------------------------------
# echo "[Step 5] Training (Recon -> Visium MSE + CDAN)"
# meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict and visualize on all samples
# Reads embeddings-hist.pickle or .npy from each P* sample via symlinks
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
# echo "[Step 6] Prediction — all samples"
# meowcat predict --config "$CFG"

echo "[Step 6] Visualization — all samples"
meowcat visualize --config "$CFG"

# ---------------------------------------------------------------------------
# Step 7: Generate PowerPoint summary
# ---------------------------------------------------------------------------
echo "[Step 7] Slide wrap"
meowcat slide --config "$CFG"

echo "============================================"
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/04_multi_visium/output/"
echo "============================================"
