"""Probe how much a single auction's price moves the model's survival prediction.

For several items we take the live snapshot, pick one listing, and sweep its
buyout across the competitor range. For each candidate price we recompute every
price-derived feature against the live competitors -- buyout_rank,
log_price_over_floor and fraction_cheaper, via the same shared helper the
training/inference pipeline uses -- run the model, and record P(sells within H)
and expected duration. If these barely move, the model is price-insensitive.

Run after retraining to compare against the previous baseline table.
"""
import os, sys, json, argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.utils import load_auctions_from_sample
from src.data.price_features import relative_price_features_for_candidate
from src.models.auction_transformer import AuctionTransformer
from src.models.inference import predict_pmf

ROOT = str(Path(__file__).resolve().parents[2])


def _load_mappings(mappings_dir):
    out = {}
    for name in ["item", "context", "bonus", "modtype"]:
        with open(os.path.join(mappings_dir, f"{name}_to_idx.json")) as f:
            out[name] = json.load(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/transformer-810.6K-survival_24-lr3e-05-bs128/last.ckpt")
    ap.add_argument("--prediction_time", default="2026-06-10 10:00:00")
    ap.add_argument("--max_hours_back", type=int, default=24)
    ap.add_argument("--n_items", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=12, help="hours for P(sale)")
    ap.add_argument("--n_prices", type=int, default=25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prediction_time = datetime.strptime(args.prediction_time, "%Y-%m-%d %H:%M:%S")
    time_left_mapping = {"VERY_LONG": 48, "LONG": 12, "MEDIUM": 2, "SHORT": 0.5}

    m = _load_mappings(os.path.join(ROOT, "generated/mappings"))
    feature_stats = torch.load(os.path.join(ROOT, "generated/feature_stats.pt"))

    df = load_auctions_from_sample(
        os.path.join(ROOT, "data/auctions/"), prediction_time, time_left_mapping,
        m["item"], m["context"], m["bonus"], m["modtype"], max_hours_back=args.max_hours_back,
    )
    model = AuctionTransformer.load_from_checkpoint(
        os.path.join(ROOT, args.ckpt), map_location=device, weights_only=False
    ).to(device)
    model.eval()

    now = df[df["snapshot_offset"] == 0]
    counts = now.groupby("item_index").size().sort_values(ascending=False)
    item_indices = counts[counts >= 5].index[: args.n_items]

    H = args.horizon
    print(f"horizon H={H}h | sweep {args.n_prices} prices per item, all price features recomputed\n")
    print(f"{'item':>7} {'n':>4} {'floor':>10} {'max':>10}  "
          f"{'P_sale @floor':>13} {'@max':>7} {'Δabs':>7}  "
          f"{'E[T] @floor':>11} {'@max':>7} {'Δh':>6}")

    for item_index in item_indices:
        df_item = df[df["item_index"] == item_index].copy()
        now_item = df_item[df_item["snapshot_offset"] == 0]
        comp = now_item["buyout"].to_numpy(dtype=float)
        floor, top = comp.min(), comp.max()
        if top <= floor:
            continue
        # Re-price the cheapest listing; score it against the other live listings.
        target_id = now_item.sort_values("buyout").iloc[0]["id"]
        competitor_buyouts = now_item.loc[~now_item["id"].eq(target_id), "buyout"].to_numpy(dtype=float)
        unique_competitors = np.unique(competitor_buyouts)
        prices = np.linspace(floor * 0.6, top * 1.4, args.n_prices)

        def _predict(price):
            d = df_item.copy()
            sel = (d["id"] == target_id) & (d["snapshot_offset"] == 0)
            log_over_floor, fraction_cheaper = relative_price_features_for_candidate(price, competitor_buyouts)
            d.loc[sel, "buyout"] = float(price)
            d.loc[sel, "buyout_rank"] = int((unique_competitors < price).sum())
            d.loc[sel, "log_price_over_floor"] = log_over_floor
            d.loc[sel, "fraction_cheaper"] = fraction_cheaper
            pmf, seq = predict_pmf(model, d, feature_stats, max_hours_back=args.max_hours_back)
            pos = np.where((seq["id"].to_numpy() == target_id) &
                           (seq["snapshot_offset"].to_numpy() == 0))[0][0]
            row = pmf[pos].numpy()
            p_sale = float(row[:H].sum())
            conditional = row / max(row.sum(), 1e-6)
            expected_duration = float((conditional * np.arange(len(row))).sum())
            return p_sale, expected_duration

        p_sale, e_dur = zip(*[_predict(p) for p in prices])
        p_sale, e_dur = np.array(p_sale), np.array(e_dur)
        print(f"{int(item_index):>7} {len(comp):>4} {floor:>10.1f} {top:>10.1f}  "
              f"{p_sale[0]:>13.3f} {p_sale[-1]:>7.3f} {abs(p_sale[0]-p_sale[-1]):>7.3f}  "
              f"{e_dur[0]:>11.2f} {e_dur[-1]:>7.2f} {abs(e_dur[0]-e_dur[-1]):>6.2f}")


if __name__ == "__main__":
    main()
