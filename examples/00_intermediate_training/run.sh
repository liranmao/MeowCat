#!/usr/bin/env bash
# =============================================================================
# Intermediate debugging run:
#   - Train on pre-prepared batch data (batch_vis_000_x/y/d.npy)
#   - Predict on P11 (01_visium_only) and P15 (01_visium_only_P15)
#   - Save all outputs under 00_intermediate_training/output/
#
# This script calls the training and prediction Python scripts DIRECTLY
# (no meowcat CLI) so we can isolate bugs in the pipeline vs the model.
# =============================================================================
set -euo pipefail
export PYTHONWARNINGS="ignore"

# ── Paths ─────────────────────────────────────────────────────────────────────
MEOWCAT_DIR="/home/liranmao/06_he_anno/code/MeowCat/MeowCat/meowcat"
TRAIN_SCRIPT="${MEOWCAT_DIR}/Train_predict/train_by_batch_cdan5_trainc_final2.py"
PREDICT_SCRIPT="${MEOWCAT_DIR}/Train_predict/predict_cdan_multireso.py"
VIS_SCRIPT="${MEOWCAT_DIR}/Train_predict/visualize_prediction_results.py"
CMAP_JSON="/home/liranmao/06_he_anno/code/MeowCat/MeowCat/config/visualization_cmap.json"

BASE="/project/KidneyHE/01_meowcat_test/00_intermediate_training"
BATCH_DIR="${BASE}/output/batches"

# Source sample data (where embeddings-hist.pickle and anno-names.txt live)
P11_SRC="/project/KidneyHE/01_meowcat_test/01_visium_only/input/VIS_P11_LUAD"
P15_SRC="/project/KidneyHE/01_meowcat_test/01_visium_only_P15/input/VIS_P15_LUAD"

# Prediction output root — prediction writes to DATA_ROOT/SAMPLE_NAME/
PRED_ROOT="${BASE}/output"

echo "============================================"
echo " Intermediate Training + Prediction"
echo " Batches: ${BATCH_DIR}"
echo " Output:  ${PRED_ROOT}"
echo "============================================"

# ── Step 0: Set up prediction output directories ──────────────────────────────
# The predict script reads anno-names.txt + embeddings-hist.pickle from
# DATA_ROOT/SAMPLE_NAME/ and saves the output .pkl there too.
# We symlink the source files so prediction outputs land in our output dir.

echo "[Step 0] Setting up prediction output directories..."

for SAMPLE_NAME in VIS_P11_LUAD VIS_P15_LUAD; do
    OUT_SAMPLE="${PRED_ROOT}/${SAMPLE_NAME}"
    mkdir -p "${OUT_SAMPLE}"

    # Determine source
    if [ "${SAMPLE_NAME}" = "VIS_P11_LUAD" ]; then
        SRC="${P11_SRC}"
    else
        SRC="${P15_SRC}"
    fi

    # Symlink anno-names.txt and embeddings-hist.pickle (needed for prediction)
    for F in anno-names.txt embeddings-hist.pickle; do
        if [ -f "${SRC}/${F}" ] && [ ! -e "${OUT_SAMPLE}/${F}" ]; then
            ln -sf "${SRC}/${F}" "${OUT_SAMPLE}/${F}"
            echo "  ${SAMPLE_NAME}/${F} -> ${SRC}/${F}"
        elif [ ! -f "${SRC}/${F}" ]; then
            echo "  WARNING: ${SRC}/${F} not found!"
        fi
    done
done

# ── Step 1: Train ─────────────────────────────────────────────────────────────
# Two-stage: Phase 0 (recon, 15 ep) -> Phase 1 (Visium MSE, 100 ep)
# This is the EXACT command that pipeline.py generates for 01_visium_only config.
echo ""
echo "[Step 1] Training (Recon 15ep -> Visium MSE 100ep)"
echo "  Batch dir: ${BATCH_DIR}"

python -u "${TRAIN_SCRIPT}" "${BATCH_DIR}" \
    --n-states 2 \
    --adv-lambda 0 \
    --freeze-encoder-n 2 \
    --recon-weight 0.1 \
    --recon-mask-ratio 0.3 \
    --save-every-n-epochs 10 \
    --xenium-weight 0.0 \
    --monitor-metric val_weak_mse \
    --device cuda \
    --two-stage --epochs1 15 \
    --epochs2 100

echo "[Step 1] Training complete. Checkpoints: ${BATCH_DIR}/states/"

# ── Step 2: Predict on P11 ───────────────────────────────────────────────────
echo ""
echo "[Step 2] Predicting on VIS_P11_LUAD"

python "${PREDICT_SCRIPT}" \
    "${BATCH_DIR}" \
    2 \
    VIS_P11_LUAD \
    --data-root "${PRED_ROOT}" \
    --device cuda \
    --tokens-per-chunk 70000 \
    --chunks-per-batch 2 \
    --out-pkl-name pred_fullgrid_outputs.pkl

# ── Step 3: Predict on P15 ───────────────────────────────────────────────────
echo ""
echo "[Step 3] Predicting on VIS_P15_LUAD"

python "${PREDICT_SCRIPT}" \
    "${BATCH_DIR}" \
    2 \
    VIS_P15_LUAD \
    --data-root "${PRED_ROOT}" \
    --device cuda \
    --tokens-per-chunk 70000 \
    --chunks-per-batch 2 \
    --out-pkl-name pred_fullgrid_outputs.pkl

# ── Step 4: Visualize P11 ─────────────────────────────────────────────────────
# --data-root  = where the prediction pkl lives (our output dir)
# --data-root-ori = where the original mask/ folder lives (source input dir)
# --out-root   = where visualization PNGs go
echo ""
echo "[Step 4] Visualizing VIS_P11_LUAD"

python "${VIS_SCRIPT}" \
    --data-root "${PRED_ROOT}" \
    --data-root-ori "${P11_SRC}/.." \
    --sample VIS_P11_LUAD \
    --pkl-name pred_fullgrid_outputs.pkl \
    --out-root "${PRED_ROOT}" \
    --n-clusters 6 \
    --pca-comp 100 \
    --random-seed 0 \
    --p-lo 5 \
    --p-hi 95 \
    --cmap-json "${CMAP_JSON}"

# ── Step 5: Visualize P15 ─────────────────────────────────────────────────────
echo ""
echo "[Step 5] Visualizing VIS_P15_LUAD"

python "${VIS_SCRIPT}" \
    --data-root "${PRED_ROOT}" \
    --data-root-ori "${P15_SRC}/.." \
    --sample VIS_P15_LUAD \
    --pkl-name pred_fullgrid_outputs.pkl \
    --out-root "${PRED_ROOT}" \
    --n-clusters 6 \
    --pca-comp 100 \
    --random-seed 0 \
    --p-lo 5 \
    --p-hi 95 \
    --cmap-json "${CMAP_JSON}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Done."
echo " Checkpoints: ${BATCH_DIR}/states/00/ and states/01/"
echo " Predictions:"
echo "   ${PRED_ROOT}/VIS_P11_LUAD/pred_fullgrid_outputs.pkl"
echo "   ${PRED_ROOT}/VIS_P15_LUAD/pred_fullgrid_outputs.pkl"
echo " Visualizations:"
echo "   ${PRED_ROOT}/VIS_P11_LUAD/*.png"
echo "   ${PRED_ROOT}/VIS_P15_LUAD/*.png"
echo "============================================"
