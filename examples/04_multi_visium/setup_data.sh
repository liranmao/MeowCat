#!/usr/bin/env bash
# =============================================================================
# Setup script: link existing preprocessed Visium data into the 04_multi_visium
# example directory structure.  Run ONCE before training.
#
# What it does:
#   1. Symlinks each P* sample folder into input/ (prediction reads from here)
#   2. Converts the old batch_000.pkl to per-domain batch_vis_XXX_x/y/d.npy
#
# NOTE: Prediction outputs (pred_fullgrid_outputs.pkl) will be written into
#       the original sample directories via the symlinks.
# =============================================================================
set -euo pipefail

# ---- Configurable paths ----------------------------------------------------
SRC_DATA="/project/KidneyHE/data_lung/7_new_sc_data"
SRC_BATCH="${SRC_DATA}/batches_visium_subset_all_sample_new_p21_out/batch_000.pkl"
DEST="/project/KidneyHE/01_meowcat_test/04_multi_visium"
EXCLUDE="P21_LUAD"
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up 04_multi_visium ==="

# 1. Create directories
mkdir -p "${DEST}/input"
mkdir -p "${DEST}/output/batches"

# 2. Symlink sample folders (P* except excluded)
echo "Linking sample folders from ${SRC_DATA} ..."
count=0
for d in "${SRC_DATA}"/P*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ "$name" = "$EXCLUDE" ]; then
        echo "  Skip: $name (excluded)"
        continue
    fi
    if [ ! -e "${DEST}/input/${name}" ]; then
        ln -s "$d" "${DEST}/input/${name}"
        echo "  Linked: ${name}"
    else
        echo "  Exists: ${name}"
    fi
    count=$((count + 1))
done
echo "  Total: $count samples"

# 3. Convert batch pickle to per-domain npy files
echo ""
echo "Converting batch pickle to per-domain npy files..."
python "${SCRIPT_DIR}/convert_pkl_to_npy.py" "$SRC_BATCH" "${DEST}/output/batches"

echo ""
echo "=== Setup complete ==="
echo "  Input:   ${DEST}/input/"
ls -1 "${DEST}/input/"
echo ""
echo "  Batches: ${DEST}/output/batches/"
ls -1 "${DEST}/output/batches/"*.npy 2>/dev/null || echo "  (no npy files found)"
