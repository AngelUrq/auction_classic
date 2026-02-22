import pytest
from scripts.transform.prepare_sequence_data import is_expired, is_sold, EXPIRED_LISTING_DURATIONS

def test_is_expired_logic_with_floats():
    """Verify is_expired correctly interprets floating parameters"""
    assert is_expired(11.0, 0.5) == 1.0
    assert is_expired(23.0, 0.0) == 1.0
    assert is_expired(47.0, 0.4) == 1.0

def test_is_expired_logic_rejects_long_durations():
    """Verify is_expired returns 0.0 if time_left is greater than 0.5"""
    assert is_expired(11.0, 2.0) == 0.0
    assert is_expired(23.0, 12.0) == 0.0
    assert is_expired(47.0, 48.0) == 0.0
    
def test_is_expired_logic_rejects_unlisted_durations():
    """Verify is_expired returns 0.0 if listing_duration is not in EXPIRED_LISTING_DURATIONS"""
    assert is_expired(10.0, 0.5) == 0.0
    assert is_expired(24.0, 0.0) == 0.0
    assert is_expired(48.0, 0.4) == 0.0

def test_is_sold_logic():
    """Verify sold correctly flags only on buyout_rank 0 and is_expired 0"""
    # Sold condition: Rank 0 + not expired
    assert is_sold(0.0, 0.0) == 1.0
    
    # Not sold conditions
    assert is_sold(1.0, 0.0) == 0.0  # Not rank 0
    assert is_sold(0.0, 1.0) == 0.0  # Expired
    assert is_sold(2.0, 1.0) == 0.0  # Not rank 0 AND expired

def test_early_snapshot_correct_labels():
    """
    Verify that an auction evaluated at listing_age=0 (e.g. current time_left=48.0)
    gets its TRUE final labels based on its final listing_duration and final time_left,
    NOT its current time_left.
    
    Example: An auction correctly completed and sold after 12 hours.
    In snapshot 1 (listing_age=0), its current time_left is 48.0.
    But its final listing_duration will be 12, and its final time_left might be 0.0 (sold).
    """
    # Final properties: it was bought after 12 hours (so it did not expire naturally)
    final_listing_duration = 12.0
    final_time_left = 12.0 # It didn't reach 0.5, so it didn't expire naturally
    
    # Evaluate expiration based on FINAL properties
    is_expired_val = is_expired(final_listing_duration, final_time_left)
    assert is_expired_val == 0.0 # It did not expire naturally
    
    # If it was the cheapest (rank 0), it should be marked as sold, 
    # even though we are evaluating it at listing_age=0
    current_buyout_rank = 0.0
    sold_val = is_sold(current_buyout_rank, is_expired_val)
    assert sold_val == 1.0
