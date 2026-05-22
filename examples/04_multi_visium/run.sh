#!/usr/bin/env bash
# =============================================================================
# Example 4a: Train on multiple Visium samples (no Xenium)
#             Multi-patient with CDAN cross-patient domain adaptation
#
# Training paradigm:
#   Phase 0 (Recon):  reconstruction pretraining              (15 epochs)
#   Phase 1 (Visium): MSE on RCTD soft proportions + CDAN     (100 epochs)
#
# CDAN aligns patient-level domains (each WSI = one domain) so the encoder
# learns patient-invariant features.
#
# Expected input layout (demo: 4 LUAD patients P11/P17/P19/P24):
#   <data_root>/
#     P11_LUAD/                      <- Visium patient
#       he_raw.tif
#       filtered_feature_bc_matrix/
#       spatial/                     <- scalefactors_json.json + tissue_positions
#       deconvolution_rctd/major_prop.csv   <- pre-computed RCTD (bundled in demo)
#       pixel-size-raw.txt           <- 0.2513 µm/px (bundled in demo)
#     P17_LUAD/  P19_LUAD/  P24_LUAD/  ... (same layout)
#
# Expected output layout:
#   <out_root>/
#     mask/
#     batches/
#       batch_vis_000_x/y/d.npy      <- domain 0 (alphabetically first)
#       batch_vis_001_x/y/d.npy      <- domain 1
#       ...
#       states/00/model.ckpt
#       states/01/model.ckpt
#     P11_LUAD/  P17_LUAD/  ...
#       embeddings-hist.pickle or .npy
#       pred_fullgrid_outputs.pkl
#     results_ex04_multi_visium.pptx
#
# Usage:
#   conda activate he_anno
#   bash run.sh > log.txt 2>&1 &
# =============================================================================
set -euo pipefail
export PYTHONWARNINGS="ignore"

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 4a: Multi-Visium training"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: RCTD deconvolution for all Visium samples
# Skipped: the Zenodo demo bundle ships deconvolution_rctd/major_prop.csv per
# Visium sample, so this step is unnecessary. For NEW samples (not the demo),
# uncomment the two lines below and set rctd.reference_rds in the config.
# ---------------------------------------------------------------------------
# echo "[Step 1] RCTD deconvolution (all Visium samples)"
# meowcat rctd --config "$CFG"

# ---------------------------------------------------------------------------
# Step 2: Resolution check
# Skipped: the Zenodo demo bundle ships pixel-size-raw.txt per sample.
# For new samples: if you know the H&E resolution, write it (in µm/px) into
# pixel-size-raw.txt in each sample folder. If you DO NOT know it and your
# image is a TIF with embedded resolution metadata, uncomment the following
# code to auto-detect.
# ---------------------------------------------------------------------------
# echo "[Step 2] Resolution check"
# meowcat check-resolution --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3: Preprocess all Visium images
# Auto-discovers all VIS* samples from project.sample_pattern.
# Can run in parallel on a cluster (one job per sample).
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 3] Image preprocessing — all Visium samples"
meowcat preprocess --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3.5: Prepare Visium metadata + embeddings
# Creates: anno-names.txt, anno_matrix.tsv, locs.tsv, radius.txt,
#          pixel-size.txt, embeddings-hist.pickle or .npy
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 3.5] Visium metadata preparation"
meowcat prepare-visium --config "$CFG"

# ---------------------------------------------------------------------------
# Step 4: Build Visium training batches
# Domain IDs are auto-assigned alphabetically (one per WSI folder) unless
# domain_map_tsv is set. For the demo: P11_LUAD=0, P17_LUAD=1, P19_LUAD=2,
# P24_LUAD=3.
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 4] Visium batch preparation"
meowcat prepare-visium-batches --config "$CFG"

# No Xenium steps (no prepare-xenium-batches needed)

# ---------------------------------------------------------------------------
# Step 5: Train — Recon + Visium MSE + CDAN
# Activate: conda activate he_anno
# Phase 0 (15 ep):   reconstruction pretraining
# Phase 1 (100 ep):  Visium MSE + CDAN domain adversarial (adv_lambda=0.005)
#   To enable out-of-sample monitoring, set train.oos_sample in the config.
# ---------------------------------------------------------------------------
echo "[Step 5] Training (Recon -> Visium MSE + CDAN)"
meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict and visualize on all Visium samples
# Activate: conda activate he_anno
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
echo " Done. Outputs: <DEMO_DATA_ROOT>/04_multi_visium/output/"
echo "============================================"
