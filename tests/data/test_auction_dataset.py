"""
Tests for src/data/auction_dataset.py :: AuctionDataset
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch

from data.auction_dataset import AuctionDataset


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_memmaps(root, data: np.ndarray, contexts=None, bonus_ids=None,
                   modifier_types=None, modifier_values=None):
    """Write raw binary memmap files to root directory."""
    N = data.shape[0]
    if contexts is None:
        contexts = np.zeros(N, dtype=np.int32)
    if bonus_ids is None:
        bonus_ids = np.zeros((N, 9), dtype=np.int32)
    if modifier_types is None:
        modifier_types = np.zeros((N, 11), dtype=np.int32)
    if modifier_values is None:
        modifier_values = np.zeros((N, 11), dtype=np.float32)

    data.astype(np.float32).tofile(root / "data.npy")
    contexts.astype(np.int32).tofile(root / "contexts.npy")
    bonus_ids.astype(np.int32).tofile(root / "bonus_ids.npy")
    modifier_types.astype(np.int32).tofile(root / "modifier_types.npy")
    modifier_values.astype(np.float32).tofile(root / "modifier_values.npy")


def _build_auction_dataset(tmp_path, data=None, idx_map=None, pairs=None,
                    feature_stats=None, max_hours_back=0):
    """
    Create a minimal AuctionDataset with 3 rows of data.
    Default: item_idx=42, one snapshot at 2025-01-10 00:00:00 covering rows 0-2.
    """
    root = tmp_path / "memmap"
    root.mkdir(exist_ok=True)

    if data is None:
        # columns: bid, buyout, quantity, time_left, listing_age, buyout_rank, is_expired, sold, listing_duration
        data = np.array([
            [100.0,  1000.0, 1.0, 12.0, 0.0, 0.0, 0.0, 0.0, 24.0],
            [200.0,  2000.0, 1.0, 48.0, 1.0, 1.0, 0.0, 0.0, 23.0],
            [ 50.0,   500.0, 2.0,  0.5, 2.0, 0.0, 1.0, 0.0, 22.0],
        ], dtype=np.float32)
    _write_memmaps(root, data)

    if idx_map is None:
        idx_map = {(42, "2025-01-10 00:00:00"): (0, data.shape[0])}

    if pairs is None:
        pairs = pd.DataFrame({
            "record": ["2025-01-10 00:00:00"],
            "item_index": [42],
        })

    return AuctionDataset(
        pairs=pairs,
        idx_map_global=idx_map,
        feature_stats=feature_stats,
        root=str(root),
        max_hours_back=max_hours_back,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_getitem_returns_all_expected_keys(tmp_path):
    """__getitem__ must return a dict with exactly the 11 documented keys."""
    ds = _build_auction_dataset(tmp_path)
    sample = ds[0]
    expected_keys = {
        "auction_features", "item_index", "contexts", "bonus_ids",
        "modifier_types", "modifier_values", "hour_of_week", "snapshot_offset",
        "listing_age", "time_left", "listing_duration", "is_expired", "sold"
    }
    assert set(sample.keys()) == expected_keys


def test_auction_features_has_5_columns(tmp_path):
    """auction_features must have shape (T, 5) — listing_duration is excluded."""
    ds = _build_auction_dataset(tmp_path)
    sample = ds[0]
    assert sample["auction_features"].shape[1] == 5


def test_listing_duration_extracted_as_y(tmp_path):
    """listing_duration tensor must match column 6 of the raw data."""
    data = np.array([
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 24.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 12.0],
    ], dtype=np.float32)
    ds = _build_auction_dataset(tmp_path, data=data)
    sample = ds[0]
    expected = torch.tensor([24.0, 12.0])
    torch.testing.assert_close(sample["listing_duration"], expected)


def test_log1p_applied_to_bid(tmp_path):
    """The bid column must be log1p-transformed before being returned."""
    data = np.array([
        [100.0, 1000.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0],
    ], dtype=np.float32)
    ds = _build_auction_dataset(tmp_path, data=data)
    sample = ds[0]
    expected_bid = float(np.log1p(100.0))
    assert sample["auction_features"][0, 0].item() == pytest.approx(expected_bid, rel=1e-5)


def test_log1p_applied_to_buyout(tmp_path):
    """The buyout column must be log1p-transformed before being returned."""
    data = np.array([
        [0.0, 500.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0],
    ], dtype=np.float32)
    ds = _build_auction_dataset(tmp_path, data=data)
    sample = ds[0]
    expected_buyout = float(np.log1p(500.0))
    assert sample["auction_features"][0, 1].item() == pytest.approx(expected_buyout, rel=1e-5)


def test_log1p_applied_to_modifier_values(tmp_path):
    """Modifier values must be log1p-transformed before being returned."""
    root = tmp_path / "memmap"
    root.mkdir(exist_ok=True)
    data = np.array([[0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0]], dtype=np.float32)
    mod_vals = np.zeros((1, 11), dtype=np.float32)
    mod_vals[0, 0] = 99.0
    _write_memmaps(root, data, modifier_values=mod_vals)

    ds = AuctionDataset(
        pairs=pd.DataFrame({"record": ["2025-01-10 00:00:00"], "item_index": [42]}),
        idx_map_global={(42, "2025-01-10 00:00:00"): (0, 1)},
        root=str(root),
    )
    sample = ds[0]
    expected = float(np.log1p(99.0))
    assert sample["modifier_values"][0, 0].item() == pytest.approx(expected, rel=1e-5)


def test_normalization_applied_when_feature_stats_given(tmp_path):
    """When feature_stats is provided, auction_features must be (x - mean) / (std + 1e-6) normalised."""
    data = np.array([[10.0, 100.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0]], dtype=np.float32)
    feature_stats = {
        "means": torch.zeros(5),
        "stds": torch.ones(5),
        "modifiers_mean": torch.tensor(0.0),
        "modifiers_std": torch.tensor(1.0),
    }
    ds = _build_auction_dataset(tmp_path, data=data, feature_stats=feature_stats)
    sample = ds[0]
    raw_bid = float(np.log1p(10.0))
    assert sample["auction_features"][0, 0].item() == pytest.approx(raw_bid / (1.0 + 1e-6), rel=1e-4)


def test_no_normalization_when_feature_stats_none(tmp_path):
    """When feature_stats is None, auction_features must only have log1p applied."""
    data = np.array([[10.0, 100.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0]], dtype=np.float32)
    ds = _build_auction_dataset(tmp_path, data=data, feature_stats=None)
    sample = ds[0]
    expected_bid = float(np.log1p(10.0))
    assert sample["auction_features"][0, 0].item() == pytest.approx(expected_bid, rel=1e-5)


def test_hour_of_week_in_valid_range(tmp_path):
    """All hour_of_week values must be in [0, 167] (7 days × 24 hours − 1)."""
    ds = _build_auction_dataset(tmp_path)
    sample = ds[0]
    how = sample["hour_of_week"]
    assert how.min().item() >= 0
    assert how.max().item() <= 167


def test_max_hours_back_zero_returns_only_current_snapshot(tmp_path):
    """With max_hours_back=0, only the rows for the current hour must be included."""
    root = tmp_path / "memmap"
    root.mkdir()
    data = np.zeros((5, 9), dtype=np.float32)
    _write_memmaps(root, data)

    idx_map = {
        (42, "2025-01-10 00:00:00"): (0, 3),
        (42, "2025-01-10 01:00:00"): (3, 2),
    }
    pairs = pd.DataFrame({"record": ["2025-01-10 00:00:00"], "item_index": [42]})

    ds = AuctionDataset(pairs=pairs, idx_map_global=idx_map,
                        root=str(root), max_hours_back=0)
    sample = ds[0]
    assert sample["auction_features"].shape[0] == 3


def test_missing_key_raises_keyerror(tmp_path):
    """Requesting a sample whose (item_idx, hour_key) is absent from idx_map must raise KeyError."""
    ds = _build_auction_dataset(tmp_path)
    extra_pair = pd.DataFrame({"record": ["2025-01-10 06:00:00"], "item_index": [99]})
    ds.pairs = pd.concat([ds.pairs, extra_pair], ignore_index=True)

    with pytest.raises(KeyError):
        _ = ds[1]

def test_is_expired_extracted_correctly(tmp_path):
    """is_expired tensor must correctly match the column indices"""
    data = np.array([
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 24.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 12.0],
    ], dtype=np.float32)
    ds = _build_auction_dataset(tmp_path, data=data)
    sample = ds[0]
    expected = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(sample["is_expired"], expected)


def test_is_sold_extracted_correctly(tmp_path):
    """sold tensor must correctly match the column indices"""
    data = np.array([
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 24.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 12.0],
    ], dtype=np.float32)
    ds = _build_auction_dataset(tmp_path, data=data)
    sample = ds[0]
    expected = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(sample["sold"], expected)
