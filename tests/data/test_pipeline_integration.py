"""
End-to-end integration test: runs all 5 stages on synthetic data and loads
AuctionDataset, verifying shapes and basic sanity properties.

Data layout:
  - 2 items (item_id=100, item_id=200), 2 auctions each
  - 3 prediction files: 2025-01-10 T00, 2025-01-10 T12, 2025-01-11 T00
  - Auction last_appearance = 2025-01-11 T12  (set via final "future" file)
    so all 3 prediction times have listing_duration >= 0;
    T00 and T12 on day 1 have listing_duration > 0.
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# Pipeline stages
from compute_timestamps import process_auctions as stage1_timestamps
from process_mappings import process_mappings as stage2_mappings
from prepare_sequence_data import process_auctions as stage3_sequences
import convert_hdf5_to_npy

from data.auction_dataset import AuctionDataset
from data.utils import collate_auctions

from conftest import (
    file_path_for,
    write_auction_file,
    make_auction,
    make_timestamps_dict,
)


# ---------------------------------------------------------------------------
# Fixture: full synthetic pipeline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """
    Runs all pipeline stages on synthetic 2-item, 3-day dataset.
    Returns a dict with paths to all generated artifacts.
    """
    root = tmp_path_factory.mktemp("integration")

    data_dir = root / "data"
    output_dir = root / "generated"
    output_dir.mkdir()
    mappings_dir = output_dir / "mappings"
    memmap_dir = output_dir / "memmap"

    # -----------------------------------------------------------------------
    # Synthetic auction data
    # -----------------------------------------------------------------------
    #   Prediction files: T00, T12 on day 1; T00 on day 2
    #   Future file:      T12 on day 2 (only for last_appearance computation)
    # -----------------------------------------------------------------------
    auctions_100 = [
        make_auction(1001, 100, 1_000_000, context=5),
        make_auction(1002, 100, 2_000_000, context=5),
    ]
    auctions_200 = [
        make_auction(2001, 200, 3_000_000, context=6),
        make_auction(2002, 200, 4_000_000, context=6),
    ]
    all_auctions = auctions_100 + auctions_200

    prediction_times = [
        datetime(2025, 1, 10, 0),
        datetime(2025, 1, 10, 12),
        datetime(2025, 1, 11, 0),
    ]
    future_time = datetime(2025, 1, 11, 12)   # sets last_appearance

    for dt in prediction_times + [future_time]:
        write_auction_file(file_path_for(data_dir, dt), all_auctions)

    # -----------------------------------------------------------------------
    # Stage 1: compute_timestamps
    # -----------------------------------------------------------------------
    ts_path = output_dir / "timestamps.json"
    timestamps = stage1_timestamps(argparse.Namespace(data_dir=str(data_dir)))
    with open(ts_path, "w") as f:
        json.dump(timestamps, f)

    # -----------------------------------------------------------------------
    # Stage 2: process_mappings
    # -----------------------------------------------------------------------
    stage2_mappings(argparse.Namespace(
        data_dir=str(data_dir),
        output_dir=str(mappings_dir),
    ))

    # -----------------------------------------------------------------------
    # Stage 3: prepare_sequence_data
    # -----------------------------------------------------------------------
    stage3_sequences(argparse.Namespace(
        data_dir=str(data_dir),
        timestamps=str(ts_path),
        mappings_dir=str(mappings_dir),
        output_dir=str(output_dir),
    ))

    # -----------------------------------------------------------------------
    # Stage 4: convert_hdf5_to_npy
    # -----------------------------------------------------------------------
    h5_path = output_dir / "sequences.h5"
    parquet_path = output_dir / "indices.parquet"
    sys.argv = [
        "convert_hdf5_to_npy",
        "--h5_path", str(h5_path),
        "--indices_path", str(parquet_path),
        "--output_dir", str(memmap_dir),
    ]
    convert_hdf5_to_npy.main()

    return {
        "root": root,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "mappings_dir": mappings_dir,
        "memmap_dir": memmap_dir,
        "h5_path": h5_path,
        "parquet_path": parquet_path,
        "prediction_times": prediction_times,
    }


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_full_pipeline_produces_loadable_dataset(pipeline_output):
    po = pipeline_output
    parquet_path = po["parquet_path"]
    memmap_dir = po["memmap_dir"]

    df = pd.read_parquet(parquet_path)
    with open(memmap_dir / "idx_map_global.pkl", "rb") as f:
        idx_map = pickle.load(f)

    ds = AuctionDataset(
        pairs=df[["record", "item_index"]],
        idx_map_global=idx_map,
        root=str(memmap_dir),
        max_hours_back=0,
    )
    assert len(ds) > 0


def test_pipeline_batch_shapes_correct(pipeline_output):
    po = pipeline_output
    df = pd.read_parquet(po["parquet_path"])
    with open(po["memmap_dir"] / "idx_map_global.pkl", "rb") as f:
        idx_map = pickle.load(f)

    ds = AuctionDataset(
        pairs=df[["record", "item_index"]],
        idx_map_global=idx_map,
        root=str(po["memmap_dir"]),
        max_hours_back=0,
    )
    loader = DataLoader(
        ds, batch_size=2,
        collate_fn=partial(collate_auctions, max_sequence_length=32),
    )
    batch = next(iter(loader))

    B = batch["auction_features"].shape[0]
    T = batch["auction_features"].shape[1]

    assert batch["auction_features"].shape == (B, T, 5)
    assert batch["bonus_ids"].shape == (B, T, 9)
    assert batch["modifier_types"].shape == (B, T, 11)
    assert batch["listing_duration"].shape == (B, T)
    assert batch["is_expired"].shape == (B, T)
    assert batch["is_sold"].shape == (B, T)


def test_pipeline_listing_duration_non_negative(pipeline_output):
    """listing_duration must never be negative (regression guard)."""
    po = pipeline_output
    df = pd.read_parquet(po["parquet_path"])
    with open(po["memmap_dir"] / "idx_map_global.pkl", "rb") as f:
        idx_map = pickle.load(f)

    ds = AuctionDataset(
        pairs=df[["record", "item_index"]],
        idx_map_global=idx_map,
        root=str(po["memmap_dir"]),
        max_hours_back=0,
    )
    loader = DataLoader(
        ds, batch_size=len(ds),
        collate_fn=partial(collate_auctions, max_sequence_length=None),
    )
    batch = next(iter(loader))
    assert batch["listing_duration"].min().item() >= 0.0
    # At least some predictions have positive duration
    assert batch["listing_duration"].max().item() > 0.0


def test_pipeline_feature_values_finite(pipeline_output):
    """No NaN or Inf in auction_features (guards against bad log1p or norm)."""
    po = pipeline_output
    df = pd.read_parquet(po["parquet_path"])
    with open(po["memmap_dir"] / "idx_map_global.pkl", "rb") as f:
        idx_map = pickle.load(f)

    ds = AuctionDataset(
        pairs=df[["record", "item_index"]],
        idx_map_global=idx_map,
        root=str(po["memmap_dir"]),
        max_hours_back=0,
    )
    loader = DataLoader(
        ds, batch_size=len(ds),
        collate_fn=partial(collate_auctions, max_sequence_length=None),
    )
    batch = next(iter(loader))
    assert torch.isfinite(batch["auction_features"]).all(), \
        "NaN or Inf found in auction_features"
    assert torch.isfinite(batch["is_expired"]).all(), \
        "NaN or Inf found in is_expired"
    assert torch.isfinite(batch["is_sold"]).all(), \
        "NaN or Inf found in is_sold"
