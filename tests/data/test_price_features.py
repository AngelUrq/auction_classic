"""
Tests for src/data/price_features.py :: relative-price features.
"""
import numpy as np
import pytest

from data.price_features import (
    compute_relative_price_features,
    relative_price_features_for_candidate,
)


def test_floor_listing_has_zero_log_and_zero_fraction():
    """The cheapest listing sits at the floor: log_price_over_floor 0, fraction_cheaper 0."""
    log_over_floor, fraction_cheaper = compute_relative_price_features([100.0, 200.0, 300.0])
    assert log_over_floor[0] == pytest.approx(0.0)
    assert fraction_cheaper[0] == pytest.approx(0.0)


def test_log_price_over_floor_is_log_of_ratio_to_cheapest():
    """log_price_over_floor must equal log(buyout / floor) for each listing."""
    log_over_floor, _ = compute_relative_price_features([100.0, 200.0, 300.0])
    np.testing.assert_allclose(log_over_floor, [0.0, np.log(2.0), np.log(3.0)], atol=1e-5)


def test_fraction_cheaper_counts_strictly_cheaper_listings():
    """fraction_cheaper must be the share of listings strictly below each price."""
    _, fraction_cheaper = compute_relative_price_features([100.0, 200.0, 300.0, 5000.0])
    np.testing.assert_allclose(fraction_cheaper, [0.0, 0.25, 0.5, 0.75], atol=1e-6)


def test_high_troll_listing_does_not_move_the_floor():
    """A huge troll listing must not change other listings' log_price_over_floor (floor = min)."""
    without_troll, _ = compute_relative_price_features([100.0, 200.0])
    with_troll, _ = compute_relative_price_features([100.0, 200.0, 1_000_000.0])
    np.testing.assert_allclose(without_troll, with_troll[:2], atol=1e-6)


def test_wall_crossing_is_captured_by_fraction_not_magnitude():
    """Listing just over a deep wall: log_price_over_floor stays tiny but fraction_cheaper approaches 1."""
    log_over_floor, fraction_cheaper = relative_price_features_for_candidate(101.0, [100.0] * 300)
    assert log_over_floor == pytest.approx(np.log(1.01), abs=1e-4)
    assert fraction_cheaper > 0.99


def test_candidate_below_all_competitors_lands_at_floor():
    """A candidate priced under every competitor becomes the floor: both features 0."""
    log_over_floor, fraction_cheaper = relative_price_features_for_candidate(99.0, [100.0, 100.0, 5000.0])
    assert log_over_floor == pytest.approx(0.0)
    assert fraction_cheaper == pytest.approx(0.0)


def test_bid_only_zero_buyout_maps_to_zero_log():
    """A bid-only listing (buyout 0) must map to log_price_over_floor 0, not -inf."""
    log_over_floor, _ = compute_relative_price_features([0.0, 100.0, 200.0])
    assert np.isfinite(log_over_floor).all()
    assert log_over_floor[0] == pytest.approx(0.0)


def test_single_listing_group_is_zero():
    """A lone listing is its own floor with nothing cheaper: both features 0."""
    log_over_floor, fraction_cheaper = compute_relative_price_features([42.0])
    assert log_over_floor[0] == pytest.approx(0.0)
    assert fraction_cheaper[0] == pytest.approx(0.0)


def test_empty_group_returns_empty_arrays():
    """An empty group must return empty feature arrays rather than erroring."""
    log_over_floor, fraction_cheaper = compute_relative_price_features([])
    assert log_over_floor.shape == (0,)
    assert fraction_cheaper.shape == (0,)
