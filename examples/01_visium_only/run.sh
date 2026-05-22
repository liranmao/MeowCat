#!/usr/bin/env bash
# =============================================================================
# Example 1: Train on a single Visium sample (soft labels, MSE loss)
#
# Training paradigm:
#   Phase 0 (Recon):  reconstruction pretraining (15 epochs)
#   Phase 1 (Visium): MSE on RCTD soft cell-type proportions (100 epochs)
#
# Expected input layout:
#   <data_root>/<SAMPLE>/
#     he_raw.tif                   <- raw H&E image
#     filtered_feature_bc_matrix/  <- 10x Space Ranger output
#     spatial/                     <- scalefactors_json.json + tissue_positions
#
# Expected output layout (created by this script):
#   <out_root>/
#     mask/                          <- HistoSweep tissue masks
#     <SAMPLE>/
#       embeddings-hist.pickle or .npy       <- UNI features [H, W, C]
#       pred_fullgrid_outputs.pkl    <- prediction results
#     batches/
#       batch_vis_000_x/y/d.npy      <- tokenized training batches
#       states/00/model.ckpt         <- trained checkpoints
#     results_ex01.pptx              <- summary slide deck
#
# Usage:
#   conda activate he_anno
#   bash run.sh > log.txt 2>&1 &
#
# All steps run inside the he_anno environment.
# =============================================================================
set -euo pipefail
export PYTHONWARNINGS="ignore"

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 1: Visium-only training"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: RCTD deconvolution (produces soft cell-type proportion labels)
# Skipped: the Zenodo demo bundle ships deconvolution_rctd/major_prop.csv per
# Visium sample, so this step is unnecessary. For NEW samples (not the demo),
# uncomment the two lines below and set rctd.reference_rds in the config.
# Output: deconvolution_rctd/major_prop.csv
# ---------------------------------------------------------------------------
# echo "[Step 1] RCTD deconvolution"
# meowcat rctd --config "$CFG"

# ---------------------------------------------------------------------------
# Step 2: Check image resolution (optional, informational only)
# Skipped: the Zenodo demo bundle ships pixel-size-raw.txt per sample.
# For new samples: if you know the H&E resolution, write it (in µm/px) into
# pixel-size-raw.txt in each sample folder. If you DO NOT know it and your
# image is a TIF with embedded resolution metadata, uncomment the following
# code to auto-detect.
# ---------------------------------------------------------------------------
# echo "[Step 2] Resolution check"
# meowcat check-resolution --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3: Preprocess the Visium image
# Activate: conda activate he_anno
# Sub-steps: get_pixel_size -> RunPreprocess -> RunHistoSweep
#            -> UNI_extract_features -> UNI_fuse_features
# Output:   he.jpg, mask/, single_super_emb.h5ad
# ---------------------------------------------------------------------------
echo "[Step 3] Image preprocessing"
meowcat preprocess --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3.5: Prepare Visium metadata + embeddings
# Activate: conda activate he_anno
# Creates: anno-names.txt, anno_matrix.tsv, locs.tsv, radius.txt,
#          pixel-size.txt, embeddings-hist.pickle or .npy
# ---------------------------------------------------------------------------
echo "[Step 3.5] Visium metadata preparation"
meowcat prepare-visium --config "$CFG"

# ---------------------------------------------------------------------------
# Step 4: Build training batches (batch_vis_*_x/y/d.npy)
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 4] Batch preparation"
meowcat prepare-visium-batches --config "$CFG"

# ---------------------------------------------------------------------------
# Step 5: Train the model
# Activate: conda activate he_anno
# Phases:  Phase 0 (recon, 15 ep) -> Phase 1 (Visium MSE, 100 ep)
# Output:  batches/states/00/model.ckpt, batches/states/01/model.ckpt
# ---------------------------------------------------------------------------
echo "[Step 5] Training (Recon -> Visium MSE)"
meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict and visualize
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
echo " Done."
echo "============================================"
