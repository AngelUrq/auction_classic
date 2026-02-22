"""
Tests for scripts/transform/convert_hdf5_to_npy.py :: main()
"""
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import convert_hdf5_to_npy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_h5(path: Path, items: dict):
    """
    items: {item_idx: np.ndarray shape (N, 7)}
    Creates a minimal sequences.h5 with the expected group structure.
    """
    with h5py.File(path, "w") as f:
        grp_root = f.create_group("items")
        for item_idx, data in items.items():
            N = data.shape[0]
            grp = grp_root.create_group(str(item_idx))
            grp.create_dataset("data", data=data.astype(np.float32))
            grp.create_dataset("contexts", data=np.zeros(N, dtype=np.int32))
            grp.create_dataset("bonus_ids", data=np.zeros((N, 9), dtype=np.int32))
            grp.create_dataset("modifier_types", data=np.zeros((N, 11), dtype=np.int32))
            grp.create_dataset("modifier_values", data=np.zeros((N, 11), dtype=np.float32))


def _build_parquet(path: Path, rows: list):
    """
    rows: list of (record, item_index, start, length)
    Writes a minimal indices.parquet.
    """
    df = pd.DataFrame(rows, columns=["record", "item_index", "start", "length"])
    # Add required stat columns (dummy values)
    for col in ["g_listing_duration_len", "g_listing_duration_mean",
                "g_listing_duration_std", "g_listing_duration_min",
                "g_listing_duration_max", "g_listing_age_mean",
                "g_listing_age_std", "g_listing_age_min", "g_listing_age_max"]:
        df[col] = 0.0
    df.to_parquet(path, index=False)


def _run_main(tmp_path, h5_path, parquet_path, output_dir, monkeypatch):
    """Monkeypatch sys.argv and call convert_hdf5_to_npy.main()."""
    monkeypatch.setattr(
        sys, "argv",
        ["convert_hdf5_to_npy",
         "--h5_path", str(h5_path),
         "--indices_path", str(parquet_path),
         "--output_dir", str(output_dir)],
    )
    convert_hdf5_to_npy.main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_memmap_files_all_created(tmp_path, monkeypatch):
    """All five .npy memmap files and idx_map_global.pkl must be created."""
    item0 = np.random.rand(5, 9).astype(np.float32)
    h5_path = tmp_path / "sequences.h5"
    parquet_path = tmp_path / "indices.parquet"
    output_dir = tmp_path / "memmap"

    _build_h5(h5_path, {42: item0})
    _build_parquet(parquet_path, [("2025-01-10 00:00:00", 42, 0, 5)])
    _run_main(tmp_path, h5_path, parquet_path, output_dir, monkeypatch)

    expected = ["data.npy", "contexts.npy", "bonus_ids.npy",
                "modifier_types.npy", "modifier_values.npy", "idx_map_global.pkl"]
    for fname in expected:
        assert (output_dir / fname).exists(), f"{fname} not created"


def test_total_rows_equals_sum_of_item_sizes(tmp_path, monkeypatch):
    """Total rows in data.npy must equal the sum of all per-item row counts."""
    item0 = np.random.rand(3, 9).astype(np.float32)
    item1 = np.random.rand(4, 9).astype(np.float32)
    h5_path = tmp_path / "sequences.h5"
    parquet_path = tmp_path / "indices.parquet"
    output_dir = tmp_path / "memmap"

    _build_h5(h5_path, {10: item0, 20: item1})
    _build_parquet(parquet_path, [
        ("2025-01-10 00:00:00", 10, 0, 3),
        ("2025-01-10 00:00:00", 20, 0, 4),
    ])
    _run_main(tmp_path, h5_path, parquet_path, output_dir, monkeypatch)

    data_mm = np.memmap(output_dir / "data.npy", mode="r", dtype=np.float32)
    total_rows = data_mm.size // 9
    assert total_rows == 7


def test_global_offset_for_each_item_correct(tmp_path, monkeypatch):
    """Items are laid out in sorted order: the first item starts at 0, the second at N_item_0."""
    item0 = np.random.rand(3, 9).astype(np.float32)
    item1 = np.random.rand(5, 9).astype(np.float32)
    h5_path = tmp_path / "sequences.h5"
    parquet_path = tmp_path / "indices.parquet"
    output_dir = tmp_path / "memmap"

    _build_h5(h5_path, {10: item0, 20: item1})
    _build_parquet(parquet_path, [
        ("2025-01-10 00:00:00", 10, 0, 3),
        ("2025-01-10 00:00:00", 20, 0, 5),
    ])
    _run_main(tmp_path, h5_path, parquet_path, output_dir, monkeypatch)

    with open(output_dir / "idx_map_global.pkl", "rb") as f:
        idx_map = pickle.load(f)

    start_10, len_10 = idx_map[(10, "2025-01-10 00:00:00")]
    start_20, len_20 = idx_map[(20, "2025-01-10 00:00:00")]

    assert start_10 == 0
    assert len_10 == 3
    assert start_20 == 3
    assert len_20 == 5


def test_data_losslessly_copied_from_hdf5(tmp_path, monkeypatch):
    """The memmap slice for an item must be byte-for-byte equal to its HDF5 data."""
    rng = np.random.default_rng(42)
    item_data = rng.random((4, 9)).astype(np.float32)

    h5_path = tmp_path / "sequences.h5"
    parquet_path = tmp_path / "indices.parquet"
    output_dir = tmp_path / "memmap"

    _build_h5(h5_path, {5: item_data})
    _build_parquet(parquet_path, [("2025-01-10 00:00:00", 5, 0, 4)])
    _run_main(tmp_path, h5_path, parquet_path, output_dir, monkeypatch)

    data_mm = np.memmap(output_dir / "data.npy", mode="r",
                        dtype=np.float32).reshape(-1, 9)
    np.testing.assert_array_equal(data_mm[:4], item_data)


def test_index_map_key_maps_to_correct_global_slice(tmp_path, monkeypatch):
    """A (item_idx, record) key in idx_map_global must map to the exact rows from HDF5."""
    rng = np.random.default_rng(7)
    item0 = rng.random((3, 9)).astype(np.float32)
    item1 = rng.random((2, 9)).astype(np.float32)

    h5_path = tmp_path / "sequences.h5"
    parquet_path = tmp_path / "indices.parquet"
    output_dir = tmp_path / "memmap"

    _build_h5(h5_path, {10: item0, 20: item1})
    _build_parquet(parquet_path, [
        ("2025-01-10 00:00:00", 10, 0, 3),
        ("2025-01-10 00:00:00", 20, 0, 2),
    ])
    _run_main(tmp_path, h5_path, parquet_path, output_dir, monkeypatch)

    with open(output_dir / "idx_map_global.pkl", "rb") as f:
        idx_map = pickle.load(f)

    data_mm = np.memmap(output_dir / "data.npy", mode="r",
                        dtype=np.float32).reshape(-1, 9)

    start1, length1 = idx_map[(20, "2025-01-10 00:00:00")]
    np.testing.assert_array_equal(data_mm[start1:start1 + length1], item1)


def test_idx_map_pkl_loadable(tmp_path, monkeypatch):
    """idx_map_global.pkl must be loadable as a dict with one entry per (item, record) pair."""
    item_data = np.ones((2, 9), dtype=np.float32)
    h5_path = tmp_path / "sequences.h5"
    parquet_path = tmp_path / "indices.parquet"
    output_dir = tmp_path / "memmap"

    _build_h5(h5_path, {3: item_data})
    _build_parquet(parquet_path, [("2025-01-10 00:00:00", 3, 0, 2)])
    _run_main(tmp_path, h5_path, parquet_path, output_dir, monkeypatch)

    pkl_path = output_dir / "idx_map_global.pkl"
    assert pkl_path.exists()
    with open(pkl_path, "rb") as f:
        idx_map = pickle.load(f)
    assert isinstance(idx_map, dict)
    assert len(idx_map) == 1
