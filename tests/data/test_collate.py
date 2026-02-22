"""
Tests for src/data/utils.py :: collate_auctions
"""
import torch
import pytest

from data.utils import collate_auctions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_item(T: int, fill=0.0) -> dict:
    """Create a batch item dict with sequence length T."""
    return {
        "auction_features":  torch.full((T, 5),  fill),
        "item_index":        torch.zeros(T, dtype=torch.int32),
        "contexts":          torch.zeros(T, dtype=torch.int32),
        "bonus_ids":         torch.zeros(T, 9,  dtype=torch.int32),
        "modifier_types":    torch.zeros(T, 11, dtype=torch.int32),
        "modifier_values":   torch.zeros(T, 11),
        "hour_of_week":      torch.zeros(T, dtype=torch.int32),
        "snapshot_offset":   torch.zeros(T, dtype=torch.int32),
        "listing_age":       torch.zeros(T),
        "time_left":         torch.zeros(T),
        "listing_duration":  torch.full((T,), fill),
        "is_expired":        torch.full((T,), fill),
        "sold":              torch.full((T,), fill),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_crop_when_under_limit():
    """A sequence shorter than max_sequence_length must not be cropped."""
    item = make_item(5)
    result = collate_auctions([item], max_sequence_length=10)
    assert result["auction_features"].shape == (1, 5, 5)


def test_crops_from_left_keeping_most_recent():
    """Cropping must keep the last L elements (most recent), discarding the oldest."""
    item = make_item(5)
    item["listing_duration"] = torch.tensor([0., 1., 2., 3., 4.])
    result = collate_auctions([item], max_sequence_length=3)

    assert result["listing_duration"].shape == (1, 3)
    torch.testing.assert_close(
        result["listing_duration"][0],
        torch.tensor([2., 3., 4.])
    )


def test_pads_shorter_sequence_to_batch_max():
    """The shorter sequence in a batch must be right-padded to the length of the longest."""
    short = make_item(3)
    long_ = make_item(5)
    result = collate_auctions([short, long_], max_sequence_length=None)

    assert result["auction_features"].shape == (2, 5, 5)


def test_pads_shorter_sequence_with_zeros():
    """Padding added to reach batch max length must be zero."""
    short = make_item(2, fill=9.0)
    long_ = make_item(4, fill=9.0)
    result = collate_auctions([short, long_], max_sequence_length=None)

    assert (result["listing_duration"][0, 2:] == 0.0).all()


def test_all_fields_have_same_first_dim():
    """All output tensors must share the same batch and time dimensions."""
    b1 = make_item(3)
    b2 = make_item(6)
    result = collate_auctions([b1, b2], max_sequence_length=None)

    T_max = 6
    for key, tensor in result.items():
        assert tensor.shape[0] == 2, f"{key}: batch dim wrong"
        assert tensor.shape[1] == T_max, f"{key}: time dim wrong, got {tensor.shape[1]}"


def test_2d_field_padded_correctly():
    """2-D fields (T, F) must be padded along the time dimension, not the feature dimension."""
    short = make_item(2)
    long_ = make_item(4)
    result = collate_auctions([short, long_])

    assert result["auction_features"].shape == (2, 4, 5)
    assert (result["auction_features"][0, 2:, :] == 0.0).all()


def test_no_limit_pads_to_longest():
    """With max_sequence_length=None the batch is padded to the longest sequence present."""
    items = [make_item(T) for T in [2, 5, 3]]
    result = collate_auctions(items, max_sequence_length=None)
    assert result["listing_duration"].shape == (3, 5)
