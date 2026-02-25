"""
Shared fixtures and path setup for data pipeline tests.
Dates used: 2025-01-10 → 2025-01-15 (safely outside exclusion window).
"""
import sys
import os

# Add scripts/transform and src to import path so tests can import pipeline modules
_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "scripts", "transform"))
sys.path.insert(0, os.path.join(_REPO, "src"))

import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper functions (not fixtures — used directly in test files)
# ---------------------------------------------------------------------------

def make_auction(auction_id, item_id, buyout, time_left="MEDIUM",
                 context=None, bid=None, quantity=1,
                 bonus_lists=None, modifiers=None, pet_species_id=None):
    """Return a minimal auction dict matching the real JSON structure."""
    item = {"id": item_id}
    if context is not None:
        item["context"] = context
    if bonus_lists is not None:
        item["bonus_lists"] = bonus_lists
    if modifiers is not None:
        item["modifiers"] = modifiers
    if pet_species_id is not None:
        item["pet_species_id"] = pet_species_id
    a = {
        "id": auction_id,
        "item": item,
        "buyout": buyout,
        "quantity": quantity,
        "time_left": time_left,
    }
    if bid is not None:
        a["bid"] = bid
    return a


def write_auction_file(path: Path, auctions: list):
    """Write {"auctions": [...]} to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"auctions": auctions}, f)


def file_path_for(base_dir: Path, dt: datetime) -> Path:
    """Return the canonical YYYY/MM/DD/YYYYMMDDThh.json path for a datetime."""
    return (
        base_dir
        / dt.strftime("%Y")
        / dt.strftime("%m")
        / dt.strftime("%d")
        / (dt.strftime("%Y%m%dT%H") + ".json")
    )


def make_minimal_mappings(output_dir: Path, item_ids=None, contexts=None,
                          bonus_ids=None, modtypes=None):
    """Write the 4 mapping JSON files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _mapping(extras):
        m = {"0": 0, "1": 1}
        idx = 2
        for x in sorted(extras or []):
            k = str(x)
            if k not in m:
                m[k] = idx
                idx += 1
        return m

    mappings = {
        "item_to_idx.json": _mapping(item_ids),
        "context_to_idx.json": _mapping(contexts),
        "bonus_to_idx.json": _mapping(bonus_ids),
        "modtype_to_idx.json": _mapping(modtypes),
    }
    for fname, mapping in mappings.items():
        with open(output_dir / fname, "w") as f:
            json.dump(mapping, f)
    return mappings


def make_timestamps_dict(auction_specs):
    """
    Build a timestamps dict {str(auction_id): {first_appearance, last_appearance, item_id, last_buyout_rank}}.

    auction_specs: list of (auction_id, item_id, first_dt, last_dt) OR (auction_id, item_id, first_dt, last_dt, last_buyout_rank)
    """
    result = {}
    for spec in auction_specs:
        if len(spec) == 5:
            auction_id, item_id, first_dt, last_dt, last_buyout_rank = spec
        else:
            auction_id, item_id, first_dt, last_dt = spec
            last_buyout_rank = 0.0

        result[str(auction_id)] = {
            "first_appearance": first_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "last_appearance": last_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "item_id": item_id,
            "last_time_left": "LONG",
            "last_buyout_rank": last_buyout_rank
        }
    return result
