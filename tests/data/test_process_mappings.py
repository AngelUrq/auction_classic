"""
Unit tests for scripts/transform/process_mappings.py :: update_mapping, process_mappings
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

from process_mappings import update_mapping, process_mappings

from conftest import file_path_for, write_auction_file, make_auction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mapping_args(data_dir, output_dir):
    return argparse.Namespace(data_dir=str(data_dir), output_dir=str(output_dir))


def _write_auction_json(data_dir, auctions, dt=None):
    if dt is None:
        dt = datetime(2025, 1, 10, 0)
    write_auction_file(file_path_for(data_dir, dt), auctions)


# ---------------------------------------------------------------------------
# update_mapping unit tests
# ---------------------------------------------------------------------------

def test_default_zero_one_entries_in_new_mapping(tmp_path):
    """A freshly created mapping must contain the default sentinel entries '0':0 and '1':1."""
    mapping = update_mapping(str(tmp_path), "item_to_idx.json", {100})
    assert mapping["0"] == 0
    assert mapping["1"] == 1


def test_new_item_added_with_next_available_index(tmp_path):
    """A new item must be assigned an index strictly greater than the default entries."""
    mapping = update_mapping(str(tmp_path), "item_to_idx.json", {100})
    assert "100" in mapping
    assert mapping["100"] >= 2


def test_incremental_update_preserves_old_indices(tmp_path):
    """Adding new items on a second call must not change existing items' indices."""
    mapping1 = update_mapping(str(tmp_path), "item_to_idx.json", {100})
    with open(tmp_path / "item_to_idx.json", "w") as f:
        json.dump(mapping1, f)
    old_idx = mapping1["100"]

    mapping2 = update_mapping(str(tmp_path), "item_to_idx.json", {200})
    assert mapping2["100"] == old_idx


def test_no_duplicate_index_values_in_mapping(tmp_path):
    """All index values in a mapping must be unique."""
    mapping = update_mapping(str(tmp_path), "item_to_idx.json", {100, 200, 300})
    values = list(mapping.values())
    assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# process_mappings integration tests
# ---------------------------------------------------------------------------

def test_all_four_files_created(tmp_path):
    """process_mappings must create all four mapping JSON files."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "mappings"
    _write_auction_json(
        data_dir,
        [make_auction(1001, 100, 1_000_000, context=5,
                      bonus_lists=[1234], modifiers=[{"type": 28, "value": 100}])],
    )
    process_mappings(_make_mapping_args(data_dir, out_dir))

    for fname in ["item_to_idx.json", "context_to_idx.json",
                  "bonus_to_idx.json", "modtype_to_idx.json"]:
        assert (out_dir / fname).exists(), f"{fname} not created"


def test_default_zero_one_entries_present_in_all_mappings(tmp_path):
    """Every mapping file must contain the sentinel entries '0':0 and '1':1."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "mappings"
    _write_auction_json(data_dir, [make_auction(1001, 100, 1_000_000)])
    process_mappings(_make_mapping_args(data_dir, out_dir))

    for fname in ["item_to_idx.json", "context_to_idx.json",
                  "bonus_to_idx.json", "modtype_to_idx.json"]:
        with open(out_dir / fname) as f:
            m = json.load(f)
        assert m.get("0") == 0, f"{fname} missing '0':0"
        assert m.get("1") == 1, f"{fname} missing '1':1"


def test_all_item_ids_in_mapping(tmp_path):
    """Every item_id present in the auction files must appear as a key in item_to_idx.json."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "mappings"
    auctions = [
        make_auction(1001, 100, 1_000_000),
        make_auction(1002, 200, 2_000_000),
        make_auction(1003, 300, 3_000_000),
    ]
    _write_auction_json(data_dir, auctions)
    process_mappings(_make_mapping_args(data_dir, out_dir))

    with open(out_dir / "item_to_idx.json") as f:
        m = json.load(f)
    for item_id in [100, 200, 300]:
        assert str(item_id) in m, f"item_id {item_id} missing from mapping"


def test_optional_fields_absent_no_crash(tmp_path):
    """Items without context, bonus_lists, or modifiers must not crash process_mappings, and the item must still be mapped."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "mappings"
    _write_auction_json(data_dir, [{"id": 9001, "item": {"id": 999},
                                   "buyout": 1000, "quantity": 1,
                                   "time_left": "MEDIUM"}])
    process_mappings(_make_mapping_args(data_dir, out_dir))  # must not raise

    with open(out_dir / "item_to_idx.json") as f:
        m = json.load(f)
    assert "999" in m
