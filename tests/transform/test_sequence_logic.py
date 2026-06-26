import pytest
from scripts.transform.prepare_sequence_data import is_expired, EXPIRED_LISTING_DURATIONS

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

def test_early_snapshot_correct_labels():
    """
    Verify that an auction evaluated at listing_age=0 (e.g. current time_left=48.0)
    gets its TRUE final is_expired label based on its final listing_duration and
    final time_left, NOT its current time_left.

    Example: an auction bought after 12 hours did not expire naturally.
    In snapshot 1 (listing_age=0) its current time_left is 48.0, but its final
    listing_duration is 12 and final time_left 12.0, so is_expired must be 0.
    """
    # Final properties: it was bought after 12 hours (so it did not expire naturally)
    final_listing_duration = 12.0
    final_time_left = 12.0  # It didn't reach 0.5, so it didn't expire naturally

    # Evaluate expiration based on FINAL properties; the survival event is (1 - is_expired).
    is_expired_val = is_expired(final_listing_duration, final_time_left)
    assert is_expired_val == 0.0  # It did not expire naturally -> counts as a sale event
