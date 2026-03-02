#!/usr/bin/env bash
# =============================================================================
# Example 2: Train on a single Xenium sample (hard labels, cross-entropy loss)
#
# Training paradigm:
#   Phase 0 (Recon):   reconstruction pretraining (15 epochs)
#   Phase 1 (Xenium):  CE on hard one-hot cell-type labels (100 epochs)
#
# Expected input layout:
#   /project/KidneyHE/01_meowcat_test/02_xenium_only/input/
#     XEN_S1/
#       he_raw.tif                        <- raw H&E image
#       adata_cellbin_HistoSweep.h5ad     <- cell-bin AnnData (with histology features)
#       XEN_S1_cell_type_anno.csv         <- per-cell manual annotations
#       anno-names.txt                    <- cell-type names (one per line)
#
# Expected output layout (created by this script):
#   /project/KidneyHE/01_meowcat_test/02_xenium_only/output/
#     mask/
#     XEN_S1/
#       embeddings-hist.pickle
#       pred_fullgrid_outputs.pkl
#     batches/
#       batch_xen_000_x/y/d.npy           <- one-hot hard labels in _y.npy
#       states/00/model.ckpt
#     results_ex02.pptx
# =============================================================================

set -euo pipefail

CFG="$(dirname "$0")/config.yaml"
# NOTE: Do NOT hardcode sample names here.
# The pipeline auto-discovers samples from project.sample_pattern in config.yaml.
# Use --samples only to override (e.g. --samples XEN_P01).

echo "============================================"
echo " MeowCat Example 2: Xenium-only training"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: RCTD — NOT needed for Xenium (hard labels come from annotations)
# ---------------------------------------------------------------------------
echo "[Step 1] Skipped — Xenium data uses hard cell-type labels, no RCTD needed"

# ---------------------------------------------------------------------------
# Step 2: Check image resolution
# Activate: micromamba activate rapids_singlecell
# ---------------------------------------------------------------------------
echo "[Step 2] Resolution check"
meowcat check-resolution --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3: Preprocess the Xenium image
# Activate: micromamba activate rapids_singlecell
# ---------------------------------------------------------------------------
echo "[Step 3] Image preprocessing"
meowcat preprocess --config "$CFG"

# ---------------------------------------------------------------------------
# Step 4: Build training batches (batch_xen_*_x/y/d.npy)
# Activate: conda activate he_anno
# Output: batches/batch_xen_000_y.npy  [N, K]  <- one-hot hard labels
# ---------------------------------------------------------------------------
echo "[Step 4] Xenium batch preparation"
meowcat prepare-xenium-batches --config "$CFG"

# ---------------------------------------------------------------------------
# Step 5: Train the model
# Activate: conda activate he_anno
# Phases:  Phase 0 (recon, 15 ep) -> Phase 1 (Xenium CE, 100 ep)
# The model operates in sc_only mode (no batch_vis_* files present).
# ---------------------------------------------------------------------------
echo "[Step 5] Training (Recon -> Xenium CE)"
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
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/02_xenium_only/output/"
echo "============================================"
