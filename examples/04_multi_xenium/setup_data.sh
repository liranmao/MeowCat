#!/usr/bin/env bash
# =============================================================================
# Setup data symlinks for Example 4b: Multi-Xenium (4 LUAD samples)
#
# Samples: P24_LUAD_Xenium, P19_LUAD_Xenium, P17_LUAD_Xenium, P11_LUAD_Xenium
#
# Creates symlinks from scattered source locations into the expected layout:
#   input/{sample}/
#     adata_cellbin_HistoSweep.h5ad  -> /project/CATCH/.../Xenium_data/{sample}/
#     xenium_raw/cell_feature_matrix.h5  -> /project/KidneyHE/.../00_luad_xenium/{sample}/{sample}/
#     xenium_raw/cells.parquet           -> /project/KidneyHE/.../00_luad_xenium/{sample}/{sample}/
#     single_super_emb.h5ad             -> /project/KidneyHE/.../00_luad_xenium/binned/{sample}/
#     annotation.csv                    -> /project/KidneyHE/.../xenium_cell_type_anno/{sample}.csv
#   input/anno-names.txt               <- 8 coarse cell types
# =============================================================================
set -euo pipefail

INPUT_DIR="/project/KidneyHE/01_meowcat_test/04_multi_xenium/input"
SAMPLES=(P24_LUAD_Xenium P19_LUAD_Xenium P17_LUAD_Xenium P11_LUAD_Xenium)

CELLBIN_ROOT="/project/CATCH/dataset/fuduanData/Xenium_data"
XENIUM_ROOT="/project/KidneyHE/data_lung/00_luad_xenium"

echo "Creating input directory: $INPUT_DIR"
mkdir -p "$INPUT_DIR"

for s in "${SAMPLES[@]}"; do
    echo "--- Setting up $s ---"
    SDIR="$INPUT_DIR/$s"
    mkdir -p "$SDIR/xenium_raw"

    # adata_cellbin_HistoSweep.h5ad
    ln -sfn "$CELLBIN_ROOT/$s/adata_cellbin_HistoSweep.h5ad" "$SDIR/adata_cellbin_HistoSweep.h5ad"

    # xenium_raw/cell_feature_matrix.h5
    ln -sfn "$XENIUM_ROOT/$s/$s/cell_feature_matrix.h5" "$SDIR/xenium_raw/cell_feature_matrix.h5"

    # xenium_raw/cells.parquet
    ln -sfn "$XENIUM_ROOT/$s/$s/cells.parquet" "$SDIR/xenium_raw/cells.parquet"

    # single_super_emb.h5ad (pre-prepared)
    ln -sfn "$XENIUM_ROOT/binned/$s/single_super_emb.h5ad" "$SDIR/single_super_emb.h5ad"

    # annotation.csv
    ln -sfn "$XENIUM_ROOT/xenium_cell_type_anno/$s.csv" "$SDIR/annotation.csv"

    echo "  Symlinks created for $s"
done

# Write anno-names.txt (8 coarse types)
ANNO_FILE="$INPUT_DIR/anno-names.txt"
echo "Writing $ANNO_FILE"
cat > "$ANNO_FILE" << 'EOF'
B
Plasma
Myeloid
Stromal
NonTumor_Epi
Tumor_Epi
T
NK
EOF

echo ""
echo "Done. Verify with: ls -la $INPUT_DIR/P*_LUAD_Xenium/"
