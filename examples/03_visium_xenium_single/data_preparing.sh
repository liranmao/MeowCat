#!/usr/bin/env bash
# =============================================================================
# Set up ex03 input: symlink raw data from ex01 (Visium) and ex02 (Xenium)
# =============================================================================
VIS_SRC=/project/KidneyHE/01_meowcat_test/01_visium_only/input/VIS_P11_LUAD
XEN_SRC=/project/KidneyHE/01_meowcat_test/wrapped_data/02_xenium_only/input/XEN_P11_LUAD
EX03=/project/KidneyHE/01_meowcat_test/03_visium_xenium_single

# Clean up previous bad symlinks
rm -f "$EX03/input/VIS_S1/he_raw*"
rmdir "$EX03/input/VIS_S1" 2>/dev/null

# Create directories
mkdir -p "$EX03/input/VIS_P11_LUAD"
mkdir -p "$EX03/input/XEN_P11_LUAD"
mkdir -p "$EX03/output"

# Visium raw inputs
ln -sfn "$VIS_SRC"/he_raw*                    "$EX03/input/VIS_P11_LUAD/"
ln -sfn "$VIS_SRC/filtered_feature_bc_matrix" "$EX03/input/VIS_P11_LUAD/"
ln -sfn "$VIS_SRC/spatial"                    "$EX03/input/VIS_P11_LUAD/"

# Xenium raw inputs
ln -sfn "$XEN_SRC"/he_raw*                       "$EX03/input/XEN_P11_LUAD/"
ln -sfn "$XEN_SRC/xenium_raw"                    "$EX03/input/XEN_P11_LUAD/"
ln -sfn "$XEN_SRC/adata_cellbin_HistoSweep.h5ad" "$EX03/input/XEN_P11_LUAD/"
ln -sfn "$XEN_SRC/annotation.csv"                "$EX03/input/XEN_P11_LUAD/"

# Verify
echo "=== VIS_P11_LUAD ==="
ls -la "$EX03/input/VIS_P11_LUAD/"
echo "=== XEN_P11_LUAD ==="
ls -la "$EX03/input/XEN_P11_LUAD/"
