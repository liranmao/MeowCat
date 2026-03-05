#!/usr/bin/env bash
# =============================================================================
# Setup symlinks for P15_LUAD test run.
# Only links the raw inputs needed to start from Step 1 (RCTD).
#
# Source: /project/KidneyHE/data_lung/P15_LUAD/
# Target: /project/KidneyHE/01_meowcat_test/01_visium_only_P15/input/VIS_P15_LUAD/
# =============================================================================
set -euo pipefail

SRC="/project/KidneyHE/data_lung/P15_LUAD"
BASE="/project/KidneyHE/01_meowcat_test/01_visium_only_P15"
INPUT="$BASE/input/VIS_P15_LUAD"

echo "Creating directory structure..."
mkdir -p "$INPUT"
mkdir -p "$BASE/output"

echo "Linking raw inputs from $SRC -> $INPUT"

# 1. Raw H&E image (pipeline expects he_raw.* via raw_flag config)
ln -sf "$SRC/00_P15_LUAD.tif" "$INPUT/he_raw.tif"

# 2. Space Ranger count matrix
ln -sf "$SRC/filtered_feature_bc_matrix" "$INPUT/filtered_feature_bc_matrix"

# 3. Space Ranger spatial metadata (tissue_positions, scalefactors)
#    NOTE: if spatial/ doesn't exist under the source, check the actual path
#    and update this line accordingly.
ln -sf "$SRC/spatial" "$INPUT/spatial"

echo ""
echo "Verifying links..."
ls -la "$INPUT/"

echo ""
echo "Done. Ready to run:"
echo "  cd $(dirname "$0") && bash run.sh > 03032026_run_P15.txt 2>&1 &"
