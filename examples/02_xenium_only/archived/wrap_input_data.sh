#!/usr/bin/env bash
# =============================================================================
# Wrap input data for Example 2 (Xenium-only P11_LUAD) into a clean directory.
#
# Creates symlinks to the original source data at:
#   /project/KidneyHE/01_meowcat_test/wrapped_data/02_xenium_only/
#
# Directory structure created:
#   wrapped_data/02_xenium_only/
#     input/XEN_P11_LUAD/
#       he_raw.tif                     -> raw H&E image
#       xenium_raw/                    -> 10x Xenium output
#       adata_cellbin_HistoSweep.h5ad  -> cell-bin location mapping (HistoSweep)
#       annotation.csv                 -> cell type annotations
#     anno-names.txt                   -> cell type order (8 coarse types)
#     output/                          -> (empty, pipeline writes here)
#
# Usage:
#   bash wrap_input_data.sh
# =============================================================================
set -euo pipefail

SAMPLE="P11_LUAD_Xenium"
XEN_SAMPLE="XEN_P11_LUAD"
WRAP="/project/KidneyHE/01_meowcat_test/wrapped_data/02_xenium_only"

# ---- Source paths ----
RAW_XENIUM="/project/KidneyHE/data_lung/00_luad_xenium/${SAMPLE}/${SAMPLE}"
CELLBIN_SRC="/project/CATCH/dataset/fuduanData/Xenium_data/${SAMPLE}"
ANNO_SRC="/project/KidneyHE/data_lung/00_luad_xenium/xenium_cell_type_anno/${SAMPLE}.csv"

# ---- Create directory structure ----
echo "Creating directory structure at ${WRAP}/ ..."
mkdir -p "$WRAP/input/${XEN_SAMPLE}"
mkdir -p "$WRAP/output"

# ---- User-provided input files (symlinks) ----
echo ""
echo "Linking input files into $WRAP/input/${XEN_SAMPLE}/ ..."

# 1. Raw H&E image
#    NOTE: Adjust the source filename if your H&E image has a different name.
HE_SRC="/project/KidneyHE/data_lung/00_luad_xenium/binned/${SAMPLE}/he.tiff"
ln -sf "$HE_SRC" \
    "$WRAP/input/${XEN_SAMPLE}/he_raw.tif"
echo "  he_raw.tif -> ${HE_SRC}"

# 2. Raw xenium data folder (cell_feature_matrix.h5, cells.parquet, etc.)
ln -sf "${RAW_XENIUM}" \
    "$WRAP/input/${XEN_SAMPLE}/xenium_raw"
echo "  xenium_raw/ -> ${RAW_XENIUM}"

# 3. Cellbin location mapping (HistoSweep alignment output)
ln -sf "${CELLBIN_SRC}/adata_cellbin_HistoSweep.h5ad" \
    "$WRAP/input/${XEN_SAMPLE}/adata_cellbin_HistoSweep.h5ad"
echo "  adata_cellbin_HistoSweep.h5ad -> ${CELLBIN_SRC}/adata_cellbin_HistoSweep.h5ad"

# 4. Cell type annotations
ln -sf "$ANNO_SRC" \
    "$WRAP/input/${XEN_SAMPLE}/annotation.csv"
echo "  annotation.csv -> ${ANNO_SRC}"

# ---- anno-names.txt (cell type order, must match training) ----
cat > "$WRAP/anno-names.txt" << 'EOF'
NonTumor_Epi
Tumor_Epi
B
Plasma
T
NK
Myeloid
Stromal
EOF
echo "  anno-names.txt (8 coarse cell types)"

# ---- (Optional) Pre-generated UNI features to skip meowcat preprocess ----
# Uncomment below to include single_super_emb.h5ad for testing without Step 3.
# For a true from-scratch run, leave this commented — Step 3 will generate it.
# HIST_SRC="/project/KidneyHE/data_lung/00_luad_xenium/binned/${SAMPLE}/single_super_emb.h5ad"
# ln -sf "$HIST_SRC" "$WRAP/input/${XEN_SAMPLE}/single_super_emb.h5ad"
# echo "  single_super_emb.h5ad -> ${HIST_SRC} (optional, for skipping Step 3)"

# ---- Verify ----
echo ""
echo "=== Wrapped data structure ==="
echo "--- ${WRAP}/ ---"
ls -la "$WRAP/"
echo "--- input/${XEN_SAMPLE}/ ---"
ls -la "$WRAP/input/${XEN_SAMPLE}/"
echo "--- anno-names.txt ---"
cat "$WRAP/anno-names.txt"
echo ""
echo "============================================"
echo " Wrapping complete."
echo " Update config.yaml paths to point to: ${WRAP}/"
echo "============================================"
