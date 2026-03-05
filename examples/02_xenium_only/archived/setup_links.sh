#!/usr/bin/env bash
# =============================================================================
# Setup symlinks for Xenium P11_LUAD test run.
#
# New unified directory structure: all data under input/XEN_P11_LUAD/
#   input/XEN_P11_LUAD/
#     xenium_raw/                    -> raw xenium data (cell_feature_matrix.h5, cells.parquet)
#     adata_cellbin_HistoSweep.h5ad  -> cellbin data
#     annotation.csv                 -> cell type annotations
#     (optional) single_super_emb.h5ad -> histology embeddings
#
# Source data:
#   Raw xenium:  /project/KidneyHE/data_lung/00_luad_xenium/P11_LUAD_Xenium/
#   Cellbin:     /project/CATCH/dataset/fuduanData/Xenium_data/P11_LUAD_Xenium/
# =============================================================================
set -euo pipefail

SAMPLE="P11_LUAD_Xenium"
XEN_SAMPLE="XEN_P11_LUAD"

# Source paths (original data)
RAW_XENIUM="/project/KidneyHE/data_lung/00_luad_xenium/${SAMPLE}"
CELLBIN_SRC="/project/CATCH/dataset/fuduanData/Xenium_data/${SAMPLE}"

# Target base
BASE="/project/KidneyHE/01_meowcat_test/02_xenium_only"
INPUT="$BASE/input/${XEN_SAMPLE}"

echo "Creating directory structure..."
mkdir -p "$INPUT"
mkdir -p "$BASE/output"

echo ""
echo "Linking raw inputs into $INPUT/ ..."

# 1. Raw xenium data: cell_feature_matrix.h5, cells.parquet
#    Pipeline expects: input/XEN_P11_LUAD/xenium_raw/
ln -sf "${RAW_XENIUM}/${SAMPLE}" "$INPUT/xenium_raw"
echo "  xenium_raw -> ${RAW_XENIUM}/${SAMPLE}"

# 2. Cellbin h5ad (with HistoSweep features)
ln -sf "${CELLBIN_SRC}/adata_cellbin_HistoSweep.h5ad" \
    "$INPUT/adata_cellbin_HistoSweep.h5ad" 2>/dev/null \
    || echo "  WARNING: adata_cellbin_HistoSweep.h5ad not found at ${CELLBIN_SRC}/"
echo "  adata_cellbin_HistoSweep.h5ad"

# 3. Cell type annotations CSV
#    Pipeline expects: input/XEN_P11_LUAD/annotation.csv
ANNO_SRC="/project/KidneyHE/data_lung/00_luad_xenium/xenium_cell_type_anno/${SAMPLE}.csv"
ln -sf "$ANNO_SRC" "$INPUT/annotation.csv" 2>/dev/null \
    || echo "  WARNING: annotation CSV not found. Please link manually to $INPUT/annotation.csv"
echo "  annotation.csv"

# 4. anno-names.txt (8 coarse cell types matching Visium order)
cat > "$BASE/anno-names.txt" << 'EOF'
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

echo ""
echo "Verifying structure..."
echo "--- input/${XEN_SAMPLE}/ ---"
ls -la "$INPUT/" 2>/dev/null || echo "  (empty)"
echo "--- anno-names.txt ---"
cat "$BASE/anno-names.txt"

echo ""
echo "============================================"
echo " Setup complete."
echo ""
echo " IMPORTANT: Verify these symlinks point to existing files."
echo " If any WARNING appeared above, fix the source paths manually."
echo ""
echo " To run:"
echo "   cd $(cd "$(dirname "$0")" && pwd) && bash run.sh > 03032026_run_xenium_P11.txt 2>&1 &"
echo "============================================"
