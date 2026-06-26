#!/usr/bin/env bash
set -euo pipefail

# Run inside the project's uv-managed environment. Override with e.g.
# PYTHON="python" bash scripts/prepare_data.sh if you manage the venv yourself.
PYTHON="${PYTHON:-uv run python}"

echo "==> [1/5] Computing timestamps..."
$PYTHON scripts/transform/compute_timestamps.py

echo "==> [2/5] Processing mappings..."
$PYTHON scripts/transform/process_mappings.py

echo "==> [3/5] Preparing sequence data (JSON -> HDF5)..."
$PYTHON scripts/transform/prepare_sequence_data.py

echo "==> [4/5] Converting HDF5 to numpy memmap..."
$PYTHON scripts/transform/convert_hdf5_to_npy.py

echo "==> [5/5] Computing feature stats..."
$PYTHON scripts/transform/compute_feature_stats.py

echo "Done."
