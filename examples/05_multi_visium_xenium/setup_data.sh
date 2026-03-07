#!/usr/bin/env bash
# =============================================================================
# Setup data for Example 5: Multi-Visium + Multi-Xenium
#
# Combines pre-prepared data from:
#   - 04_multi_visium: Visium samples (P* dirs + batch_vis_*_x/y/d.npy)
#   - 04_multi_xenium: Xenium samples (P*_LUAD_Xenium dirs)
#
# Creates:
#   input/P001/, P002/, ...           <- symlinks to 04_multi_visium Visium samples
#   input/P*_LUAD_Xenium/             <- symlinks to 04_multi_xenium Xenium samples
#   output/batches/batch_vis_*.npy   <- copied from 04_multi_visium batches
# =============================================================================
set -euo pipefail

VIS_INPUT="/project/KidneyHE/01_meowcat_test/04_multi_visium/input"
VIS_BATCHES="/project/KidneyHE/01_meowcat_test/04_multi_visium/output/batches"
XEN_INPUT="/project/KidneyHE/01_meowcat_test/04_multi_xenium/input"

DEST="/project/KidneyHE/01_meowcat_test/05_multi_visium_xenium"

echo "=== Setting up 05_multi_visium_xenium ==="

mkdir -p "${DEST}/input"
mkdir -p "${DEST}/output/batches"

# 1. Symlink Visium sample folders from 04_multi_visium
echo ""
echo "--- Linking Visium samples from ${VIS_INPUT} ---"
for d in "${VIS_INPUT}"/P*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    ln -sfn "$(readlink -f "$d")" "${DEST}/input/${name}"
    echo "  Linked: ${name}"
done

# 2. Copy Visium batch npy files (so Xenium batch prep can read them for domain offset)
echo ""
echo "--- Copying Visium batch files from ${VIS_BATCHES} ---"
for f in "${VIS_BATCHES}"/batch_vis_*.npy; do
    [ -f "$f" ] || continue
    cp -v "$f" "${DEST}/output/batches/"
done

# 3. Symlink Xenium sample folders from 04_multi_xenium
echo ""
echo "--- Linking Xenium samples from ${XEN_INPUT} ---"
for d in "${XEN_INPUT}"/P*_LUAD_Xenium/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    ln -sfn "$(readlink -f "$d")" "${DEST}/input/${name}"
    echo "  Linked: ${name}"
done

# 4. Symlink anno-names.txt from 04_multi_xenium (shared cell type list)
echo ""
echo "--- Linking anno-names.txt ---"
ln -sfn "${XEN_INPUT}/anno-names.txt" "${DEST}/input/anno-names.txt"

echo ""
echo "=== Setup complete ==="
echo "  Input samples:"
ls -1 "${DEST}/input/"
echo ""
echo "  Visium batches:"
ls -1 "${DEST}/output/batches/"batch_vis_*.npy 2>/dev/null || echo "  (none)"
