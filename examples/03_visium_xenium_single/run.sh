#!/usr/bin/env bash
# =============================================================================
# Example 3: Train on one Visium + one Xenium sample (3-phase sequential)
#
# Training paradigm:
#   Phase 0 (Recon):   reconstruction pretraining        (15 epochs)
#   Phase 1 (Visium):  MSE on RCTD soft proportions      (100 epochs)
#   Phase 2 (Xenium):  CE fine-tuning on hard labels     (50 epochs)
#
# The Xenium phase fine-tunes the encoder trained on Visium, transferring
# spot-level spatial knowledge to single-cell resolution.
#
# Expected input layout:
#   /project/KidneyHE/01_meowcat_test/03_visium_xenium_single/input/
#     VIS_S1/
#       he_raw.tif
#       filtered_feature_bc_matrix/
#       spatial/
#       anno-names.txt                 <- must have SAME cell types as Xenium
#       radius.txt
#     XEN_S1/
#       he_raw.tif
#       adata_cellbin_HistoSweep.h5ad
#       XEN_S1_cell_type_anno.csv
#       anno-names.txt                 <- must have SAME cell types as Visium
#
# Expected output layout:
#   /project/KidneyHE/01_meowcat_test/03_visium_xenium_single/output/
#     mask/
#     batches/
#       batch_vis_000_x/y/d.npy        <- Visium batches (resolution=0, MSE)
#       batch_xen_000_x/y/d.npy        <- Xenium batches (resolution=1, CE)
#       states/00/model.ckpt
#       states/01/model.ckpt
#     VIS_S1/  XEN_S1/
#       embeddings-hist.pickle
#       pred_fullgrid_outputs.pkl
#     results_ex03.pptx
# =============================================================================

set -euo pipefail

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 3: Visium + Xenium (single pair)"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: RCTD deconvolution for Visium
# Activate: conda activate RCTD
# ---------------------------------------------------------------------------
echo "[Step 1] RCTD deconvolution (Visium only)"
echo "  NOTE: Edit Preprocess/RCTD_deconvolution.R with your paths, then run:"
echo "  meowcat rctd --config $CFG"
# meowcat rctd --config "$CFG"

# ---------------------------------------------------------------------------
# Step 2: Resolution check (checks all samples matching sample_pattern)
# Activate: micromamba activate rapids_singlecell
# ---------------------------------------------------------------------------
echo "[Step 2] Resolution check"
meowcat check-resolution --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3a: Preprocess Visium image
# ---------------------------------------------------------------------------
echo "[Step 3a] Image preprocessing — VIS_S1"
meowcat preprocess --config "$CFG" --samples VIS_S1

# ---------------------------------------------------------------------------
# Step 3b: Preprocess Xenium image
# (same preprocessing steps; --samples overrides sample_pattern in config)
# ---------------------------------------------------------------------------
echo "[Step 3b] Image preprocessing — XEN_S1"
meowcat preprocess --config "$CFG" --samples XEN_S1

# ---------------------------------------------------------------------------
# Step 4: Build training batches
# Both batch_vis_* and batch_xen_* will be written to the same batches/ dir.
# The training script identifies data type from the filename prefix.
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 4] Batch preparation (Visium + Xenium)"
meowcat prepare-batches --config "$CFG"

# ---------------------------------------------------------------------------
# Step 5: Train the model — 3-phase sequential
# Activate: conda activate he_anno
# Phase 0 (15 ep):  MSE reconstruction pretraining on masked UNI features
# Phase 1 (100 ep): Visium MSE — learns soft cell-type distributions
# Phase 2 (50 ep):  Xenium CE  — fine-tunes to hard single-cell labels
#   encoder is frozen for first freeze_encoder_n=2 layers at each transition
# ---------------------------------------------------------------------------
echo "[Step 5] Training (Recon -> Visium -> Xenium)"
meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict on both samples
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 6] Prediction — VIS_S1"
meowcat predict --config "$CFG" --samples VIS_S1

echo "[Step 6] Prediction — XEN_S1"
meowcat predict --config "$CFG" --samples XEN_S1

echo "[Step 6] Visualization — both samples"
meowcat visualize --config "$CFG" --samples VIS_S1,XEN_S1

# ---------------------------------------------------------------------------
# Step 7: Generate PowerPoint summary
# ---------------------------------------------------------------------------
echo "[Step 7] Slide wrap"
meowcat slide --config "$CFG"

echo "============================================"
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/03_visium_xenium_single/output/"
echo "============================================"
