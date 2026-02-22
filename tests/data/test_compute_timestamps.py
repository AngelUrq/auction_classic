"""
Unit tests for scripts/transform/compute_timestamps.py :: process_auctions
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

from compute_timestamps import process_auctions

from conftest import file_path_for, write_auction_file, make_auction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamp_args(data_dir):
    return argparse.Namespace(data_dir=str(data_dir))


def _write_auction_json(base, dt, auction_list):
    write_auction_file(file_path_for(base, dt), auction_list)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_first_appearance_is_earliest_file(tmp_path):
    """An auction seen in two files must record the earlier timestamp as first_appearance."""
    t1 = datetime(2025, 1, 10, 0)
    t2 = datetime(2025, 1, 10, 12)
    _write_auction_json(tmp_path, t1, [make_auction(1001, 100, 0)])
    _write_auction_json(tmp_path, t2, [make_auction(1001, 100, 0)])

    result = process_auctions(_make_timestamp_args(tmp_path))

    assert result[1001]["first_appearance"] == "2025-01-10 00:00:00"


def test_last_appearance_is_latest_file(tmp_path):
    """An auction seen in three files must record the latest timestamp as last_appearance."""
    t1 = datetime(2025, 1, 10, 0)
    t2 = datetime(2025, 1, 10, 6)
    t3 = datetime(2025, 1, 10, 12)
    for t in [t1, t2, t3]:
        _write_auction_json(tmp_path, t, [make_auction(1001, 100, 0)])

    result = process_auctions(_make_timestamp_args(tmp_path))

    assert result[1001]["last_appearance"] == "2025-01-10 12:00:00"


def test_single_file_first_equals_last(tmp_path):
    """An auction seen in exactly one file must have first_appearance == last_appearance."""
    _write_auction_json(tmp_path, datetime(2025, 1, 10, 0), [make_auction(1001, 100, 0)])

    result = process_auctions(_make_timestamp_args(tmp_path))

    assert result[1001]["first_appearance"] == result[1001]["last_appearance"]


def test_item_id_stored(tmp_path):
    """The item_id from the auction JSON must be recorded in the timestamps entry."""
    _write_auction_json(tmp_path, datetime(2025, 1, 10, 0), [make_auction(1001, 42, 0)])

    result = process_auctions(_make_timestamp_args(tmp_path))

    assert result[1001]["item_id"] == 42


def test_malformed_json_skipped(tmp_path):
    """A file with invalid JSON must be skipped without crashing; valid auctions still processed."""
    t1 = datetime(2025, 1, 10, 0)
    t2 = datetime(2025, 1, 10, 12)
    _write_auction_json(tmp_path, t1, [make_auction(2001, 100, 0)])

    bad = file_path_for(tmp_path, t2)
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not valid json {{{")

    result = process_auctions(_make_timestamp_args(tmp_path))  # must not raise

    assert 2001 in result


def test_missing_auctions_key_skipped(tmp_path):
    """A JSON file without an 'auctions' key must be skipped without crashing."""
    t1 = datetime(2025, 1, 10, 0)
    t2 = datetime(2025, 1, 10, 12)
    _write_auction_json(tmp_path, t1, [make_auction(3001, 100, 0)])

    no_key = file_path_for(tmp_path, t2)
    no_key.parent.mkdir(parents=True, exist_ok=True)
    with open(no_key, "w") as f:
        json.dump({"not_auctions": True}, f)

    result = process_auctions(_make_timestamp_args(tmp_path))  # must not raise

    assert 3001 in result
