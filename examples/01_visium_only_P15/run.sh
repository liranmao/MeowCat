#!/usr/bin/env bash
# =============================================================================
# Example 1b: Train on a single Visium sample P15_LUAD (soft labels, MSE loss)
#
# Training paradigm:
#   Phase 0 (Recon):  reconstruction pretraining (15 epochs)
#   Phase 1 (Visium): MSE on RCTD soft cell-type proportions (100 epochs)
#
# Expected input layout:
#   /project/KidneyHE/01_meowcat_test/01_visium_only_P15/input/
#     VIS_P15_LUAD/
#       he_raw.tif                   <- symlink to 00_P15_LUAD.tif
#       filtered_feature_bc_matrix/  <- symlink to 10x Space Ranger output
#       spatial/                     <- symlink to scalefactors + tissue_positions
# =============================================================================
set -euo pipefail
export PYTHONWARNINGS="ignore"

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 1b: Visium-only P15_LUAD"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: RCTD deconvolution (produces soft cell-type proportion labels)
# Activate: conda activate he_anno (with Rscript symlinked)
# ---------------------------------------------------------------------------
# echo "[Step 1] RCTD deconvolution"
# meowcat rctd --config "$CFG"

# ---------------------------------------------------------------------------
# Step 2: Check image resolution
# Activate: micromamba activate rapids_singlecell
# ---------------------------------------------------------------------------
# echo "[Step 2] Resolution check"
# meowcat check-resolution --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3: Preprocess the Visium image
# Activate: micromamba activate rapids_singlecell
# Sub-steps: get_pixel_size -> RunPreprocess -> RunHistoSweep
#            -> UNI_extract_features -> UNI_fuse_features
# ---------------------------------------------------------------------------
# echo "[Step 3] Image preprocessing"
# meowcat preprocess --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3.5: Prepare Visium metadata + embeddings
# Creates: anno-names.txt, anno_matrix.tsv, locs.tsv, radius.txt,
#          pixel-size.txt, embeddings-hist.pickle
# ---------------------------------------------------------------------------
# echo "[Step 3.5] Visium metadata preparation"
# meowcat prepare-visium --config "$CFG"

# ---------------------------------------------------------------------------
# Step 4: Build training batches (batch_vis_*_x/y/d.npy)
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
# echo "[Step 4] Batch preparation"
# meowcat prepare-visium-batches --config "$CFG"

# ---------------------------------------------------------------------------
# Step 5: Train the model
# Activate: conda activate he_anno
# Phases:  Phase 0 (recon, 15 ep) -> Phase 1 (Visium MSE, 100 ep)
# ---------------------------------------------------------------------------
# echo "[Step 5] Training (Recon -> Visium MSE)"
# meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict and visualize on the training sample itself
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 6] Prediction"
meowcat predict --config "$CFG"

echo "[Step 6] Visualization"
meowcat visualize --config "$CFG"

# ---------------------------------------------------------------------------
# Step 7: Generate PowerPoint summary
# ---------------------------------------------------------------------------
echo "[Step 7] Slide wrap"
meowcat slide --config "$CFG"

echo "============================================"
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/01_visium_only_P15/output/"
echo "============================================"
