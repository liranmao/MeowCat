#!/usr/bin/env bash
# =============================================================================
# Example 1: Train on a single Visium sample (soft labels, MSE loss)
#
# Training paradigm:
#   Phase 0 (Recon):  reconstruction pretraining (15 epochs)
#   Phase 1 (Visium): MSE on RCTD soft cell-type proportions (100 epochs)
#
# Expected input layout:
#   /project/KidneyHE/01_meowcat_test/01_visium_only/input/
#     VIS_S1/
#       he_raw.tif                   <- raw H&E image
#       filtered_feature_bc_matrix/  <- 10x Space Ranger output
#       spatial/                     <- spot coordinates
#       anno-names.txt               <- cell-type names (one per line)
#       radius.txt                   <- spot radius in raw pixels
#
# Expected output layout (created by this script):
#   /project/KidneyHE/01_meowcat_test/01_visium_only/output/
#     mask/                          <- HistoSweep tissue masks
#     VIS_S1/
#       embeddings-hist.pickle       <- UNI features [H, W, C]
#       pred_fullgrid_outputs.pkl    <- prediction results
#     batches/
#       batch_vis_000_x/y/d.npy      <- tokenized training batches
#       states/00/model.ckpt         <- trained checkpoints
#     results_ex01.pptx              <- summary slide deck
# =============================================================================

set -euo pipefail

CFG="$(dirname "$0")/config.yaml"
SAMPLE="VIS_S1"

echo "============================================"
echo " MeowCat Example 1: Visium-only training"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: RCTD deconvolution (produces soft cell-type proportion labels)
# Activate: conda activate RCTD
# ---------------------------------------------------------------------------
echo "[Step 1] RCTD deconvolution"
echo "  NOTE: Edit Preprocess/RCTD_deconvolution.R with your reference atlas path"
echo "  Then run: meowcat rctd --config $CFG"
# meowcat rctd --config "$CFG"

# ---------------------------------------------------------------------------
# Step 2: Check image resolution
# Activate: micromamba activate rapids_singlecell
# ---------------------------------------------------------------------------
echo "[Step 2] Resolution check"
meowcat check-resolution --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3: Preprocess the Visium image
# Activate: micromamba activate rapids_singlecell
# Sub-steps: get_pixel_size -> RunPreprocess -> RunHistoSweep
#            -> UNI_extract_features -> UNI_fuse_features
# ---------------------------------------------------------------------------
echo "[Step 3] Image preprocessing for $SAMPLE"
meowcat preprocess --config "$CFG" --samples "$SAMPLE"

# ---------------------------------------------------------------------------
# Step 4: Build training batches (batch_vis_*_x/y/d.npy)
# Activate: conda activate he_anno
# Output: batches/batch_vis_000_x.npy  [N, T, C]
#         batches/batch_vis_000_y.npy  [N, K]  <- RCTD soft proportions
#         batches/batch_vis_000_d.npy  [N]     <- domain IDs (all 0 for single sample)
# ---------------------------------------------------------------------------
echo "[Step 4] Batch preparation"
meowcat prepare-batches --config "$CFG"

# ---------------------------------------------------------------------------
# Step 5: Train the model
# Activate: conda activate he_anno
# Phases:  Phase 0 (recon, 15 ep) -> Phase 1 (Visium MSE, 100 ep)
# Output:  batches/states/00/model.ckpt
#          batches/states/01/model.ckpt
# ---------------------------------------------------------------------------
echo "[Step 5] Training (Recon -> Visium MSE)"
meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict and visualize on the training sample itself
# (In practice, prediction runs on held-out slides)
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 6] Prediction"
meowcat predict --config "$CFG" --samples "$SAMPLE"

echo "[Step 6] Visualization"
meowcat visualize --config "$CFG" --samples "$SAMPLE"

# ---------------------------------------------------------------------------
# Step 7: Generate PowerPoint summary
# ---------------------------------------------------------------------------
echo "[Step 7] Slide wrap"
meowcat slide --config "$CFG"

echo "============================================"
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/01_visium_only/output/"
echo "============================================"
