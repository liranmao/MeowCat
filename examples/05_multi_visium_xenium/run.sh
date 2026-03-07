#!/usr/bin/env bash
# =============================================================================
# Example 5: Train on multiple Visium + multiple Xenium samples
#            (3-phase sequential + CDAN cross-patient domain adaptation)
#            Visium: P* samples from 04_multi_visium
#            Xenium: P24, P19, P17, P11 LUAD samples from 04_multi_xenium
#
# Prerequisites:
#   - Run setup_data.sh first to symlink data from 04_multi_visium and 04_multi_xenium
#   - All data is pre-prepared (Steps 1-4a skipped)
#
# Training paradigm:
#   Phase 0 (Recon):   reconstruction pretraining        (15 epochs)
#   Phase 1 (Visium):  MSE on RCTD soft proportions      (100 epochs)
#                      + CDAN adversarial loss (adv_lambda=0.005)
#   Phase 2 (Xenium):  CE fine-tuning on hard labels     (100 epochs)
#
# Input layout (created by setup_data.sh):
#   /project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/input/
#     P*/                            <- Visium samples (symlinked from 04_multi_visium)
#     P*_LUAD_Xenium/                <- Xenium samples (symlinked from 04_multi_xenium)
#
# Output layout:
#   /project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/output/
#     batches/
#       batch_vis_*_x/y/d.npy        <- Visium domains (copied by setup_data.sh)
#       batch_xen_*_x/y/d.npy        <- Xenium domains (Step 4b)
#       states/00/model.ckpt
#       states/01/model.ckpt
#     results_ex05.pptx
# =============================================================================
export PYTHONWARNINGS="ignore"
set -euo pipefail

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 5: Multi-sample Visium + Xenium"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Steps 1-4a: SKIPPED
# - Steps 1-3: all data pre-prepared (RCTD, preprocess, prepare-visium done)
# - Step 4a: Visium batches copied by setup_data.sh
# ---------------------------------------------------------------------------
echo "[Steps 1-4a] Skipped — using pre-prepared data from 04_multi_visium + 04_multi_xenium"

# ---------------------------------------------------------------------------
# Step 4b: Build Xenium training batches
# Xenium domain IDs continue after Visium (reads batch_vis_* for domain offset).
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 4b] Xenium batch preparation"
meowcat prepare-xenium-batches --config "$CFG"

# ---------------------------------------------------------------------------
# Step 5: Train — 3-phase + CDAN
# Activate: conda activate he_anno
# Phase 0 (15 ep):   reconstruction pretraining
# Phase 1 (100 ep):  Visium MSE + CDAN domain adversarial (adv_lambda=0.005)
# Phase 2 (100 ep):  Xenium CE fine-tuning (xenium_weight=0.01)
# ---------------------------------------------------------------------------
echo "[Step 5] Training (Recon -> Visium+CDAN -> Xenium)"
meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict and visualize on all samples
# Activate: conda activate he_anno
# Auto-discovers both P* (Visium) and P*_LUAD_Xenium samples from data_root.
# ---------------------------------------------------------------------------
echo "[Step 6] Prediction — all samples"
meowcat predict --config "$CFG"

echo "[Step 6] Visualization — all samples"
meowcat visualize --config "$CFG"

# ---------------------------------------------------------------------------
# Step 7: Generate PowerPoint summary
# ---------------------------------------------------------------------------
echo "[Step 7] Slide wrap"
meowcat slide --config "$CFG"

echo "============================================"
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/output/"
echo "============================================"
