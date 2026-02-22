#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/6] Computing timestamps..."
python scripts/transform/compute_timestamps.py

echo "==> [2/6] Processing mappings..."
python scripts/transform/process_mappings.py

echo "==> [3/6] Preparing sequence data (JSON -> HDF5)..."
python scripts/transform/prepare_sequence_data.py

echo "==> [4/6] Converting HDF5 to numpy memmap..."
python scripts/transform/convert_hdf5_to_npy.py

echo "==> [5/6] Computing feature stats..."
python scripts/transform/compute_feature_stats.py

echo "==> [6/6] Zipping generated folder..."
zip -r generated.zip generated/

echo "Done."
