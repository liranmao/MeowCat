#!/usr/bin/env bash
# =============================================================================
# Example 6: Predict cell types on new H&E images using a trained model
#
# Prerequisites:
#   - A trained MeowCat model (from examples 01-04 or your own training run)
#   - New H&E images placed under data_root/<SAMPLE>/he_raw.<ext>
#
# This runs the full inference pipeline:
#   1. Preprocess each sample (pixel size, rescale, mask, UNI features, fusion)
#   2. Prepare embeddings (single_super_emb.h5ad -> embeddings-hist.pickle or .npy)
#   3. Predict cell-type distributions using trained model checkpoints
#   4. Visualize results (argmax map + intensity maps)
#   5. Generate PowerPoint summary
#
# NOTE: Preprocessing (step 1) requires rapids_singlecell environment.
#       Steps 2-5 require he_anno environment.
#       Use --start-from 6 to skip preprocessing if already done.
#
# Usage:
#   # Full pipeline (all steps):
#   conda activate he_anno
#   bash run.sh > log.txt 2>&1 &
#
#   # Skip preprocessing (already preprocessed):
#   meowcat infer --config config.yaml --start-from 6
#
#   # Specific samples only:
#   meowcat infer --config config.yaml --samples HE001,HE002
# =============================================================================
set -euo pipefail
export PYTHONWARNINGS="ignore"

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 6: Predict on new H&E"
echo " Config: $CFG"
echo "============================================"

meowcat infer --config "$CFG"

echo "============================================"
echo " Done."
echo "============================================"
