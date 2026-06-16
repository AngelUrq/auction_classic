#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/5] Computing timestamps..."
python scripts/transform/compute_timestamps.py

echo "==> [2/5] Processing mappings..."
python scripts/transform/process_mappings.py

echo "==> [3/5] Preparing sequence data (JSON -> HDF5)..."
python scripts/transform/prepare_sequence_data.py

echo "==> [4/5] Converting HDF5 to numpy memmap..."
python scripts/transform/convert_hdf5_to_npy.py

echo "==> [5/5] Computing feature stats..."
python scripts/transform/compute_feature_stats.py

echo "Done."
