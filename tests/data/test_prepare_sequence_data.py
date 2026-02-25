"""
Tests for scripts/transform/prepare_sequence_data.py
Covers pure helper functions and the full process_auctions integration.
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import pytest

from prepare_sequence_data import (
    pad_or_truncate_bonuses,
    pad_or_truncate_modifiers,
    TIME_LEFT_TO_INT,
    MAX_BONUSES,
    MAX_MODIFIERS,
    process_auctions as prepare_process_auctions,
)
from conftest import (
    file_path_for,
    write_auction_file,
    make_auction,
    make_minimal_mappings,
    make_timestamps_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bonus_map(*ids):
    m = {"0": 0, "1": 1}
    for i, bid in enumerate(ids, start=2):
        m[str(bid)] = i
    return m


def _modtype_map(*types):
    m = {"0": 0, "1": 1}
    for i, t in enumerate(types, start=2):
        m[str(t)] = i
    return m


def _make_sequence_args(data_dir, timestamps_file, mappings_dir, output_dir):
    return argparse.Namespace(
        data_dir=str(data_dir),
        timestamps=str(timestamps_file),
        mappings_dir=str(mappings_dir),
        output_dir=str(output_dir),
    )


def _write_timestamps(path: Path, specs):
    ts = make_timestamps_dict(specs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ts, f)


def _build_single_auction_dir(tmp_path, auction_id=1001, item_id=100,
                           prediction_dt=None, first_dt=None, last_dt=None,
                           buyout=1_000_000, context=5):
    """
    Write one auction JSON file + timestamps + mappings for a single auction.
    Returns (data_dir, timestamps_file, mappings_dir, output_dir).
    """
    if prediction_dt is None:
        prediction_dt = datetime(2025, 1, 10, 0)
    if first_dt is None:
        first_dt = prediction_dt
    if last_dt is None:
        last_dt = datetime(2025, 1, 11, 0)

    data_dir = tmp_path / "data"
    write_auction_file(
        file_path_for(data_dir, prediction_dt),
        [make_auction(auction_id, item_id, buyout, context=context)],
    )

    ts_file = tmp_path / "timestamps.json"
    _write_timestamps(ts_file, [(auction_id, item_id, first_dt, last_dt)])

    mappings_dir = tmp_path / "mappings"
    make_minimal_mappings(mappings_dir, item_ids=[item_id], contexts=[context])

    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return data_dir, ts_file, mappings_dir, output_dir


# ---------------------------------------------------------------------------
# pad_or_truncate_bonuses
# ---------------------------------------------------------------------------

def test_pad_or_truncate_bonuses_pads_short():
    """Input shorter than MAX_BONUSES must be zero-padded to MAX_BONUSES."""
    bonus_map = _bonus_map(10, 20)
    result = pad_or_truncate_bonuses([10, 20], bonus_map)
    assert result.shape == (MAX_BONUSES,)
    assert result[0] == bonus_map["10"]
    assert result[1] == bonus_map["20"]
    assert all(result[2:] == 0)


def test_pad_or_truncate_bonuses_truncates_long():
    """Input longer than MAX_BONUSES must be truncated to exactly MAX_BONUSES."""
    ids = list(range(1, 13))  # 12 bonus IDs
    m = {"0": 0, "1": 1}
    for i, bid in enumerate(ids, start=2):
        m[str(bid)] = i
    result = pad_or_truncate_bonuses(ids, m)
    assert result.shape == (MAX_BONUSES,)


# ---------------------------------------------------------------------------
# pad_or_truncate_modifiers
# ---------------------------------------------------------------------------

def test_pad_or_truncate_modifiers_pads():
    """Input shorter than MAX_MODIFIERS must zero-pad both types and values."""
    modtype_map = _modtype_map(28, 29)
    mods = [{"type": 28, "value": 100}, {"type": 29, "value": 200}]
    types, values = pad_or_truncate_modifiers(mods, modtype_map)
    assert types.shape == (MAX_MODIFIERS,)
    assert values.shape == (MAX_MODIFIERS,)
    assert types[0] == modtype_map["28"]
    assert values[1] == 200.0
    assert all(types[2:] == 0)
    assert all(values[2:] == 0.0)


def test_pad_or_truncate_modifiers_truncates():
    """Input longer than MAX_MODIFIERS must be truncated to exactly MAX_MODIFIERS."""
    m = {"0": 0, "1": 1}
    mods = [{"type": i, "value": float(i)} for i in range(2, 15)]  # 13 modifiers
    for i in range(2, 15):
        m[str(i)] = i
    types, values = pad_or_truncate_modifiers(mods, m)
    assert types.shape == (MAX_MODIFIERS,)
    assert values.shape == (MAX_MODIFIERS,)


# ---------------------------------------------------------------------------
# TIME_LEFT_TO_INT
# ---------------------------------------------------------------------------

def test_time_left_mapping_short():
    """SHORT must map to 0.5 hours."""
    assert TIME_LEFT_TO_INT["SHORT"] == 0.5


def test_time_left_mapping_all_values():
    """All four time_left labels must map to their expected float values."""
    assert TIME_LEFT_TO_INT["VERY_LONG"] == 48.0
    assert TIME_LEFT_TO_INT["LONG"] == 12.0
    assert TIME_LEFT_TO_INT["MEDIUM"] == 2.0
    assert TIME_LEFT_TO_INT["SHORT"] == 0.5


# ---------------------------------------------------------------------------
# Feature computation formulas
# ---------------------------------------------------------------------------

def test_copper_to_gold_conversion():
    """Dividing copper by 10,000 must yield the correct gold value."""
    copper = 18_500_000
    gold = float(copper) / 10_000.0
    assert gold == 1850.0


def test_listing_age_is_hours_since_first_appearance():
    """listing_age = (prediction_time - first_appearance).total_seconds() / 3600."""
    first = datetime(2025, 1, 10, 0)
    prediction = datetime(2025, 1, 10, 12)
    listing_age = (prediction - first).total_seconds() / 3600.0
    assert listing_age == 12.0


def test_listing_duration_is_hours_from_first_to_last_appearance():
    """listing_duration = (last_appearance - first_appearance).total_seconds() / 3600."""
    first = datetime(2025, 1, 10, 0)
    last = datetime(2025, 1, 11, 0)
    listing_duration = (last - first).total_seconds() / 3600.0
    assert listing_duration == 24.0


# ---------------------------------------------------------------------------
# Buyout rank (searchsorted logic)
# ---------------------------------------------------------------------------

def test_buyout_rank_searchsorted():
    """searchsorted gives ranks 0, 1, 2 for three distinct sorted prices."""
    prices = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    unique_sorted = np.sort(np.unique(prices))
    ranks = np.searchsorted(unique_sorted, prices)
    assert ranks[0] == 0
    assert ranks[1] == 1
    assert ranks[2] == 2


def test_buyout_rank_ties_get_same_rank():
    """Two auctions with the same price must receive the same rank."""
    prices = np.array([100.0, 200.0, 100.0], dtype=np.float32)
    unique_sorted = np.sort(np.unique(prices))
    ranks = np.searchsorted(unique_sorted, prices)
    assert ranks[0] == ranks[2]
    assert ranks[1] > ranks[0]


# ---------------------------------------------------------------------------
# process_auctions integration tests
# ---------------------------------------------------------------------------

def test_process_auctions_produces_h5_and_parquet(tmp_path):
    """Running process_auctions must create both sequences.h5 and indices.parquet."""
    data_dir, ts_file, mappings_dir, output_dir = _build_single_auction_dir(tmp_path)
    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    assert (output_dir / "sequences.h5").exists()
    assert (output_dir / "indices.parquet").exists()


def test_h5_item_group_data_shape(tmp_path):
    """Each item group in sequences.h5 must have a data dataset with shape (N, 7)."""
    data_dir, ts_file, mappings_dir, output_dir = _build_single_auction_dir(
        tmp_path, item_id=100, context=5
    )
    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        data = h5f[f"items/{item_index}/data"][:]
    assert data.ndim == 2
    assert data.shape[1] == 9


def test_indices_parquet_schema(tmp_path):
    """indices.parquet must contain all 13 expected columns."""
    data_dir, ts_file, mappings_dir, output_dir = _build_single_auction_dir(tmp_path)
    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    df = pd.read_parquet(output_dir / "indices.parquet")
    expected_cols = [
        "record", "item_index",
        "g_listing_duration_len", "g_listing_duration_mean", "g_listing_duration_std",
        "g_listing_duration_min", "g_listing_duration_max",
        "g_listing_age_mean", "g_listing_age_std",
        "g_listing_age_min", "g_listing_age_max",
        "start", "length",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' missing from parquet"


def test_data_has_9_columns_in_h5(tmp_path):
    """The data dataset inside each HDF5 item group must have exactly 9 columns."""
    data_dir, ts_file, mappings_dir, output_dir = _build_single_auction_dir(tmp_path)
    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        shape = h5f[f"items/{item_index}/data"].shape
    assert shape[1] == 9


def test_buyout_rank_is_column_5_in_output(tmp_path):
    """Column 5 of data must contain buyout ranks (0, 1, 2), not raw prices."""
    prediction_dt = datetime(2025, 1, 10, 0)
    last_dt = datetime(2025, 1, 11, 0)
    data_dir = tmp_path / "data"
    auctions = [
        make_auction(1001, 100, 1_000_000, context=5),
        make_auction(1002, 100, 2_000_000, context=5),
        make_auction(1003, 100, 3_000_000, context=5),
    ]
    write_auction_file(file_path_for(data_dir, prediction_dt), auctions)

    ts_file = tmp_path / "timestamps.json"
    _write_timestamps(ts_file, [
        (1001, 100, prediction_dt, last_dt),
        (1002, 100, prediction_dt, last_dt),
        (1003, 100, prediction_dt, last_dt),
    ])
    mappings_dir = tmp_path / "mappings"
    make_minimal_mappings(mappings_dir, item_ids=[100], contexts=[5])
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        data = h5f[f"items/{item_index}/data"][:]

    assert set(data[:, 5].tolist()) == {0.0, 1.0, 2.0}


def test_listing_duration_is_column_8_in_output(tmp_path):
    """Column 8 of data must equal (last_appearance - first_appearance) in hours."""
    first_dt = datetime(2025, 1, 10, 0)
    prediction_dt = datetime(2025, 1, 10, 6)   # 6 h after first — must NOT affect duration
    last_dt = datetime(2025, 1, 11, 0)          # 24 h after first

    data_dir, ts_file, mappings_dir, output_dir = _build_single_auction_dir(
        tmp_path, prediction_dt=prediction_dt, first_dt=first_dt, last_dt=last_dt
    )
    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        data = h5f[f"items/{item_index}/data"][:]

    assert data[0, 8] == pytest.approx(24.0, abs=0.01)


def test_is_expired_is_column_6_in_output(tmp_path):
    """Column 6 of data must contain the is_expired label."""
    first_dt = datetime(2025, 1, 10, 0)
    prediction_dt = datetime(2025, 1, 10, 6)
    last_dt = datetime(2025, 1, 10, 23)  # exactly 23 hours duration
    
    data_dir, ts_file, mappings_dir, output_dir = _build_single_auction_dir(
        tmp_path, prediction_dt=prediction_dt, first_dt=first_dt, last_dt=last_dt
    )
    
    ts = {
        "1001": {
            "first_appearance": first_dt.strftime("%Y-%m-%d %H:%M:%S"), 
            "last_appearance": last_dt.strftime("%Y-%m-%d %H:%M:%S"), 
            "item_id": 100, 
            "last_time_left": "SHORT",
            "last_buyout_rank": 0.0
        }
    }
    with open(ts_file, "w") as f: 
        json.dump(ts, f)

    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        data = h5f[f"items/{item_index}/data"][:]

    # Custom time left is SHORT (0.5) 
    # For a duration of 23.0 (which evaluates to EXPIRED_LISTING_DURATIONS) and time_left 0.5,
    # the function will return 1.0.
    assert data[0, 6] == pytest.approx(1.0, abs=0.01)


def test_is_sold_is_column_7_in_output(tmp_path):
    """Column 7 of data must contain the sold label."""
    first_dt = datetime(2025, 1, 10, 0)
    prediction_dt = datetime(2025, 1, 10, 6)
    last_dt = datetime(2025, 1, 10, 23)  # 23 hours duration so it's expired
    
    data_dir, ts_file, mappings_dir, output_dir = _build_single_auction_dir(
        tmp_path, prediction_dt=prediction_dt, first_dt=first_dt, last_dt=last_dt,
        buyout=1_000_000, auction_id=1001
    )
    # Add a cheaper auction to ensure buyout_rank > 0 (it will not sell)
    write_auction_file(
        file_path_for(data_dir, prediction_dt),
        [make_auction(1001, 100, 1_000_000, context=5), make_auction(1002, 100, 500_000, context=5)]
    )
    # Update mock timestamps with new auction
    ts = {"1001": {"first_appearance": first_dt.strftime("%Y-%m-%d %H:%M:%S"), "last_appearance": last_dt.strftime("%Y-%m-%d %H:%M:%S"), "item_id": 100, "last_time_left": "SHORT"},
          "1002": {"first_appearance": first_dt.strftime("%Y-%m-%d %H:%M:%S"), "last_appearance": last_dt.strftime("%Y-%m-%d %H:%M:%S"), "item_id": 100, "last_time_left": "SHORT"}}
    with open(ts_file, "w") as f: json.dump(ts, f)

    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        data = h5f[f"items/{item_index}/data"][:]

    # data[0] is auction 1001 (buyout_rank=1), data[1] is auction 1002 (buyout_rank=0)
    # For auction 1001, buyout_rank=1, so sold=0.0
    assert data[0, 7] == pytest.approx(0.0, abs=0.01)


def test_pet_auctions_excluded(tmp_path):
    """Auctions with pet_species_id in item must not appear in the HDF5 output."""
    prediction_dt = datetime(2025, 1, 10, 0)
    last_dt = datetime(2025, 1, 11, 0)

    data_dir = tmp_path / "data"
    auctions = [
        make_auction(1001, 100, 1_000_000, context=5),
        make_auction(1002, 100, 2_000_000, context=5, pet_species_id=123),
    ]
    write_auction_file(file_path_for(data_dir, prediction_dt), auctions)

    ts_file = tmp_path / "timestamps.json"
    _write_timestamps(ts_file, [
        (1001, 100, prediction_dt, last_dt),
        (1002, 100, prediction_dt, last_dt),
    ])
    mappings_dir = tmp_path / "mappings"
    make_minimal_mappings(mappings_dir, item_ids=[100], contexts=[5])
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        n_rows = h5f[f"items/{item_index}/data"].shape[0]

    assert n_rows == 1


def test_sequence_capped_at_1024(tmp_path):
    """More than 1024 auctions for one item/hour must be capped to exactly 1024 rows."""
    prediction_dt = datetime(2025, 1, 10, 0)
    last_dt = datetime(2025, 1, 11, 0)

    data_dir = tmp_path / "data"
    n_auctions = 1025
    start_id = 5000
    auctions = [
        make_auction(start_id + i, 100, 1_000_000 + i * 100, context=5)
        for i in range(n_auctions)
    ]
    write_auction_file(file_path_for(data_dir, prediction_dt), auctions)

    ts_file = tmp_path / "timestamps.json"
    specs = [(start_id + i, 100, prediction_dt, last_dt) for i in range(n_auctions)]
    _write_timestamps(ts_file, specs)

    mappings_dir = tmp_path / "mappings"
    make_minimal_mappings(mappings_dir, item_ids=[100], contexts=[5])
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        n_rows = h5f[f"items/{item_index}/data"].shape[0]

    assert n_rows == 1024


def test_excluded_dates_not_in_parquet(tmp_path):
    """Records from hardcoded excluded dates must not appear in indices.parquet."""
    good_dt = datetime(2025, 1, 10, 0)
    excluded_dt = datetime(2025, 12, 20, 0)
    last_dt = datetime(2025, 1, 11, 0)

    data_dir = tmp_path / "data"
    write_auction_file(
        file_path_for(data_dir, good_dt),
        [make_auction(1001, 100, 1_000_000, context=5)],
    )
    write_auction_file(
        file_path_for(data_dir, excluded_dt),
        [make_auction(2001, 200, 2_000_000, context=6)],
    )

    ts_file = tmp_path / "timestamps.json"
    _write_timestamps(ts_file, [
        (1001, 100, good_dt, last_dt),
        (2001, 200, excluded_dt, excluded_dt),
    ])

    mappings_dir = tmp_path / "mappings"
    make_minimal_mappings(mappings_dir, item_ids=[100, 200], contexts=[5, 6])
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    df = pd.read_parquet(output_dir / "indices.parquet")
    records = df["record"].tolist()

    assert any("2025-01-10" in r for r in records)
    assert not any("2025-12-20" in r for r in records)


def test_is_sold_is_consistent_across_snapshots(tmp_path):
    """An auction that eventually sells (is_sold=1.0) must have that label in EVERY snapshot, even when temporarily undercut."""
    first_dt = datetime(2025, 1, 10, 0)
    mid_dt = datetime(2025, 1, 10, 1)
    last_dt = datetime(2025, 1, 10, 2)
    
    data_dir = tmp_path / "data"
    
    # Hour 0: Auction 1001 is cheapest (Rank 0)
    write_auction_file(
        file_path_for(data_dir, first_dt),
        [make_auction(1001, 100, 1_000_000, context=5)]
    )
    
    # Hour 1: Auction 1002 undercuts (Auction 1001 becomes Rank 1)
    write_auction_file(
        file_path_for(data_dir, mid_dt),
        [
            make_auction(1001, 100, 1_000_000, context=5),
            make_auction(1002, 100, 500_000, context=5)
        ]
    )
    
    # Hour 2: Auction 1002 disappears. Auction 1001 is cheapest again (Rank 0). This is 1001's final snapshot.
    write_auction_file(
        file_path_for(data_dir, last_dt),
        [make_auction(1001, 100, 1_000_000, context=5)]
    )

    ts_file = tmp_path / "timestamps.json"
    _write_timestamps(ts_file, [
        (1001, 100, first_dt, last_dt, 0.0), 
        (1002, 100, mid_dt, mid_dt, 1.0)     
    ])

    mappings_dir = tmp_path / "mappings"
    make_minimal_mappings(mappings_dir, item_ids=[100], contexts=[5])
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    prepare_process_auctions(_make_sequence_args(data_dir, ts_file, mappings_dir, output_dir))

    with open(mappings_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    item_index = item_to_idx["100"]

    with h5py.File(output_dir / "sequences.h5", "r") as h5f:
        data = h5f[f"items/{item_index}/data"][:]

    # Filter out Auction 1002's data point, we only care about 1001's 3 snapshots
    # (Since there are 4 rows total in `data`, 3 for 1001, 1 for 1002. 
    # Row indices in H5 are: 0 (hour 0: 1001), 1 (hour 1: 1001), 2 (hour 1: 1002), 3 (hour 2: 1001)
    # The prices are in column 1.
    is_1001 = data[:, 1] == 100.0  # 1_000_000 / 10000.0 = 100.0
    data_1001 = data[is_1001]
    
    assert data_1001.shape[0] == 3
    
    # The sold column is column 7. It should be 1.0 for ALL THREE snapshots because 
    # the auction ultimately disappeared while Rank 0.
    assert data_1001[0, 7] == pytest.approx(1.0, abs=0.01) # Hour 0
    assert data_1001[1, 7] == pytest.approx(1.0, abs=0.01) # Hour 1 (currently fails because rank is 1)
    assert data_1001[2, 7] == pytest.approx(1.0, abs=0.01) # Hour 2
