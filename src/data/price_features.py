import numpy as np


def compute_relative_price_features(buyouts):
    """Per snapshot x item relative-price features for one item's listings.

    Given the buyout prices of every listing of one item at one snapshot, return
    two per-listing features that describe where each price sits in the order book:

      log_price_over_floor: log(buyout / floor), where floor is the cheapest
          positive buyout in the group. 0 at the floor, log(2) at twice the floor,
          and so on. Encodes how far above the cheapest competitor a listing is.
          Immune to high troll listings (they do not move the floor).

      fraction_cheaper: fraction of listings strictly cheaper than this one, in
          [0, 1). 0 for the cheapest, ~1 for a price sitting above the whole book.
          Ordinal, so immune to troll listings on either side.

    Listings with buyout <= 0 (bid-only) map to log_price_over_floor 0.

    Returns:
        (log_price_over_floor, fraction_cheaper): two float32 arrays aligned to
        the input order.
    """
    buyouts = np.asarray(buyouts, dtype=np.float64)
    n = buyouts.shape[0]
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    positive_buyouts = buyouts[buyouts > 0]
    floor = float(positive_buyouts.min()) if positive_buyouts.size else 1.0

    log_price_over_floor = np.where(
        buyouts > 0, np.log(np.maximum(buyouts, 1e-9) / floor), 0.0
    )

    sorted_buyouts = np.sort(buyouts)
    fraction_cheaper = np.searchsorted(sorted_buyouts, buyouts, side="left") / n

    return log_price_over_floor.astype(np.float32), fraction_cheaper.astype(np.float32)


def relative_price_features_for_candidate(candidate_buyout, competitor_buyouts):
    """Relative-price features for a single candidate price within a competitor set.

    The candidate is included in the group (matching how training computes the
    features over all listings), so a price below every competitor lands at the
    floor (log_price_over_floor 0, fraction_cheaper 0). Used by the recommendation
    sweep when re-pricing a relisting against the live competitors.

    Returns:
        (log_price_over_floor, fraction_cheaper) as Python floats for the candidate.
    """
    buyouts = np.concatenate([
        np.asarray(competitor_buyouts, dtype=np.float64),
        np.asarray([candidate_buyout], dtype=np.float64),
    ])
    log_price_over_floor, fraction_cheaper = compute_relative_price_features(buyouts)
    return float(log_price_over_floor[-1]), float(fraction_cheaper[-1])
