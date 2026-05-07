#!/usr/bin/env bash
# =============================================================================
# Example 4a: Train on multiple Visium samples (no Xenium) — FROM SCRATCH
#             Multi-patient with CDAN cross-patient domain adaptation
#
# Training paradigm:
#   Phase 0 (Recon):  reconstruction pretraining              (15 epochs)
#   Phase 1 (Visium): MSE on RCTD soft proportions + CDAN     (100 epochs)
#
# CDAN aligns patient-level domains (each WSI = one domain) so the encoder
# learns patient-invariant features. VIS_S2 is held out as an OOS monitor.
#
# Expected input layout:
#   /project/KidneyHE/01_meowcat_test/04_multi_visium/input/
#     VIS_S1/                        <- Visium patient 1
#       he_raw.tif
#       filtered_feature_bc_matrix/
#       spatial/                     <- scalefactors_json.json + tissue_positions
#     VIS_S2/                        <- Visium patient 2 (also OOS monitor)
#       ... (same layout as VIS_S1)
#
# Expected output layout:
#   /project/KidneyHE/01_meowcat_test/04_multi_visium/output/
#     mask/
#     batches/
#       batch_vis_000_x/y/d.npy      <- VIS_S1 (domain 0)
#       batch_vis_001_x/y/d.npy      <- VIS_S2 (domain 1)
#       oos_batch/                   <- OOS monitoring batches for VIS_S2
#       states/00/model.ckpt
#       states/01/model.ckpt
#     VIS_S1/  VIS_S2/
#       embeddings-hist.pickle or .npy
#       pred_fullgrid_outputs.pkl
#     results_ex04_multi_visium.pptx
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
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 1] RCTD deconvolution (all Visium samples)"
meowcat rctd --config "$CFG"

# ---------------------------------------------------------------------------
# Step 2: Resolution check
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 2] Resolution check"
meowcat check-resolution --config "$CFG"

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
# Domain IDs are auto-assigned (one per WSI folder) unless domain_map_tsv is set.
#   domain 0 -> VIS_S1, domain 1 -> VIS_S2
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
#   VIS_S2 is monitored as out-of-sample via --oos-sample (no gradient leak)
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
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/04_multi_visium/output/"
echo "============================================"
