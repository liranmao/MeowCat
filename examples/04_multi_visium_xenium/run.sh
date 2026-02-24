#!/usr/bin/env bash
# =============================================================================
# Example 4: Train on multiple Visium + multiple Xenium samples
#            (3-phase sequential + CDAN cross-patient domain adaptation)
#
# Training paradigm:
#   Phase 0 (Recon):   reconstruction pretraining        (15 epochs)
#   Phase 1 (Visium):  MSE on RCTD soft proportions      (100 epochs)
#                      + CDAN adversarial loss (adv_lambda=0.005)
#   Phase 2 (Xenium):  CE fine-tuning on hard labels     (100 epochs)
#
# CDAN aligns patient-level domains (each WSI = one domain) so the encoder
# learns patient-invariant features.
# VIS_S2 is held out as an out-of-sample monitor during training.
#
# Expected input layout:
#   /project/KidneyHE/01_meowcat_test/04_multi_visium_xenium/input/
#     VIS_S1/                        <- Visium patient 1
#       he_raw.tif
#       filtered_feature_bc_matrix/
#       spatial/
#       anno-names.txt
#       radius.txt
#     VIS_S2/                        <- Visium patient 2 (also OOS monitor)
#       ... (same layout as VIS_S1)
#     XEN_S1/                        <- Xenium patient 1
#       he_raw.tif
#       adata_cellbin_HistoSweep.h5ad
#       XEN_S1_cell_type_anno.csv
#       anno-names.txt
#     XEN_S2/                        <- Xenium patient 2
#       ... (same layout as XEN_S1)
#
# Expected output layout:
#   /project/KidneyHE/01_meowcat_test/04_multi_visium_xenium/output/
#     mask/
#     batches/
#       batch_vis_000_x/y/d.npy      <- VIS_S1 (domain 0)
#       batch_vis_001_x/y/d.npy      <- VIS_S2 (domain 1)
#       batch_xen_000_x/y/d.npy      <- XEN_S1 (domain 2)
#       batch_xen_001_x/y/d.npy      <- XEN_S2 (domain 3)
#       oos_batch/                   <- OOS monitoring batches for VIS_S2
#       states/00/model.ckpt
#       states/01/model.ckpt
#     VIS_S1/  VIS_S2/  XEN_S1/  XEN_S2/
#       embeddings-hist.pickle
#       pred_fullgrid_outputs.pkl
#     results_ex04.pptx
# =============================================================================

set -euo pipefail

CFG="$(dirname "$0")/config.yaml"

echo "============================================"
echo " MeowCat Example 4: Multi-sample Visium + Xenium"
echo " Config: $CFG"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: RCTD deconvolution for both Visium samples
# Activate: conda activate RCTD
# ---------------------------------------------------------------------------
echo "[Step 1] RCTD deconvolution (Visium samples)"
echo "  NOTE: Edit Preprocess/RCTD_deconvolution.R to loop over VIS_S1 and VIS_S2"
echo "  Then run: meowcat rctd --config $CFG"
# meowcat rctd --config "$CFG"

# ---------------------------------------------------------------------------
# Step 2: Resolution check
# Activate: micromamba activate rapids_singlecell
# ---------------------------------------------------------------------------
echo "[Step 2] Resolution check"
meowcat check-resolution --config "$CFG"

# ---------------------------------------------------------------------------
# Step 3: Preprocess all four samples
# Can run in parallel on a cluster (one job per sample).
# Activate: micromamba activate rapids_singlecell
# ---------------------------------------------------------------------------
echo "[Step 3] Image preprocessing — all 4 samples"
for SAMPLE in VIS_S1 VIS_S2 XEN_S1 XEN_S2; do
    echo "  -> $SAMPLE"
    meowcat preprocess --config "$CFG" --samples "$SAMPLE"
done

# ---------------------------------------------------------------------------
# Step 4: Build training batches for all samples
# Domain IDs are auto-assigned (one per WSI folder) unless domain_map_tsv is set.
#   domain 0 -> VIS_S1, domain 1 -> VIS_S2
#   domain 2 -> XEN_S1, domain 3 -> XEN_S2
# The CDAN adversarial head will align across all 4 domains.
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 4] Batch preparation (4 samples)"
meowcat prepare-batches --config "$CFG"

# ---------------------------------------------------------------------------
# Step 5: Train — 3-phase + CDAN
# Activate: conda activate he_anno
# Phase 0 (15 ep):   reconstruction pretraining
# Phase 1 (100 ep):  Visium MSE + CDAN domain adversarial (adv_lambda=0.005)
#   VIS_S2 is monitored as out-of-sample via --oos-sample (no gradient leak)
# Phase 2 (100 ep):  Xenium CE fine-tuning (xenium_weight=0.01)
# ---------------------------------------------------------------------------
echo "[Step 5] Training (Recon -> Visium+CDAN -> Xenium)"
meowcat train --config "$CFG"

# ---------------------------------------------------------------------------
# Step 6: Predict and visualize on all samples
# Activate: conda activate he_anno
# ---------------------------------------------------------------------------
echo "[Step 6] Prediction — all samples"
for SAMPLE in VIS_S1 VIS_S2 XEN_S1 XEN_S2; do
    echo "  -> predict $SAMPLE"
    meowcat predict --config "$CFG" --samples "$SAMPLE"
done

echo "[Step 6] Visualization — all samples"
meowcat visualize --config "$CFG" --samples VIS_S1,VIS_S2,XEN_S1,XEN_S2

# ---------------------------------------------------------------------------
# Step 7: Generate PowerPoint summary
# ---------------------------------------------------------------------------
echo "[Step 7] Slide wrap"
meowcat slide --config "$CFG"

echo "============================================"
echo " Done. Outputs: /project/KidneyHE/01_meowcat_test/04_multi_visium_xenium/output/"
echo "============================================"
