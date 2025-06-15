import torch
import h5py
import numpy as np
from datetime import datetime

class AuctionDataset(torch.utils.data.Dataset):
    """Loads auction sequences and injects engineered + calendar features.

    Numeric feature layout per auction row after this update (17 columns):
        0   bid              (raw gold)
        1   buyout           (raw gold)
        2   quantity
        3   time_left        (hours)
        4   current_hours    (age)
        5   hour_sin         sin(2π·hour/24)
        6   weekday_sin      sin(2π·weekday/7)
        7   hour_cos         cos(2π·hour/24)
        8   is_weekend       1 if weekday∈{5,6} else 0
        9   delta_low        (price−min)/min
       10   delta_high       (p95−price)/p95
       11   unit_price       buyout/quantity
       12   unit_rank        dense rank of unit_price
       13   spread           (p95−min)/min
       14   num_auctions     snapshot supply
       15   median_price     snapshot median
       16   lowtail_vol      #auctions ≤1.05·min
    """

    def __init__(self, pairs, feature_stats=None, path="../generated/sequences.h5"):
        self.pairs = pairs
        self.feature_stats = feature_stats

        self.column_map = {
            "bid": 0,
            "buyout": 1,
            "quantity": 2,
            "time_left": 3,
            "current_hours": 4,
            "hour_sin": 5,
            "weekday_sin": 6,
            "hour_cos": 7,
            "is_weekend": 8,
            "delta_low": 9,
            "delta_high": 10,
            "unit_price": 11,
            "unit_rank": 12,
            "spread": 13,
            "num_auctions": 14,
            "median_price": 15,
            "lowtail_vol": 16,
        }

        self.h5_file = h5py.File(path, "r")
        print(f"Dataset size: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __del__(self):
        self.h5_file.close()

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        ts = datetime.strptime(pair["record"], "%Y-%m-%d %H:%M:%S")
        item_idx = pair["item_index"]
        grp_path = f"{ts:%Y-%m-%d}/{ts:%H}/{item_idx}"
        g = self.h5_file[grp_path]

        auctions = torch.tensor(g["auctions"][:], dtype=torch.float32)
        contexts = torch.tensor(g["contexts"][:], dtype=torch.int32)
        bonus_lists = torch.tensor(g["bonus_lists"][:], dtype=torch.int32)
        modifier_types = torch.tensor(g["modifier_types"][:], dtype=torch.int32)
        modifier_values = torch.tensor(g["modifier_values"][:], dtype=torch.float32)
        buyout_rank = torch.tensor(g["buyout_ranking"][:], dtype=torch.int32)

        hours_on_sale = auctions[:, -1]
        auctions = auctions[:, :-1]

        buyouts = auctions[:, 1].numpy()
        quantities = auctions[:, 2].numpy()
        valid_mask = buyouts > 0
        min_b = buyouts[valid_mask].min() if valid_mask.any() else 0.0
        p95_b = np.percentile(buyouts[valid_mask], 95) if valid_mask.any() else min_b
        median_b = np.median(buyouts[valid_mask]) if valid_mask.any() else min_b

        delta_low = (buyouts - min_b) / (min_b + 1e-6)
        delta_high = (p95_b - buyouts) / (p95_b + 1e-6)
        unit_price = buyouts / np.maximum(quantities, 1)
        rank_order = unit_price.argsort(kind="mergesort")
        unit_rank = np.empty_like(rank_order, dtype=np.float32)
        unit_rank[rank_order] = np.arange(1, len(unit_price) + 1)
        spread = (p95_b - min_b) / (min_b + 1e-6)
        num_auc = float(len(buyouts))
        lowtail_vol = float((buyouts <= 1.05 * min_b).sum())

        eng_feats = np.column_stack([
            delta_low,
            delta_high,
            unit_price,
            unit_rank,
            np.full_like(buyouts, spread,      dtype=np.float32),
            np.full_like(buyouts, num_auc,     dtype=np.float32),
            np.full_like(buyouts, median_b,    dtype=np.float32),
            np.full_like(buyouts, lowtail_vol, dtype=np.float32),
        ])

        # Calendar placeholders (hour_sin, weekday_sin, hour_cos, is_weekend)
        auctions = torch.cat([
            auctions,                                      # 5 original features
            torch.zeros((auctions.size(0), 4)),            # temporal slots
            torch.tensor(eng_feats, dtype=torch.float32)   # 9 engineered
        ], dim=1)

        # Temporal features
        hour_angle     = 2 * np.pi * ts.hour / 24
        auctions[:, self.column_map["hour_sin"]]  = np.sin(hour_angle)
        auctions[:, self.column_map["hour_cos"]]  = np.cos(hour_angle)
        weekday_angle  = 2 * np.pi * ts.weekday() / 7
        auctions[:, self.column_map["weekday_sin"]] = np.sin(weekday_angle)
        auctions[:, self.column_map["is_weekend"]] = 1.0 if ts.weekday() >= 5 else 0.0

        # Log‑scale bid + buyout
        auctions[:, self.column_map["bid"]]    = torch.log1p(auctions[:, self.column_map["bid"]])
        auctions[:, self.column_map["buyout"]] = torch.log1p(auctions[:, self.column_map["buyout"]])
        modifier_values = torch.log1p(modifier_values)

        # Normalize
        if self.feature_stats is not None:
            auctions = (auctions - self.feature_stats["means"]) / (self.feature_stats["stds"] + 1e-6)
            modifier_values = (modifier_values - self.feature_stats["modifiers_mean"]) / (self.feature_stats["modifiers_std"] + 1e-6)

        item_index_tensor = torch.tensor(item_idx, dtype=torch.int32).repeat(auctions.size(0))
        current_hours = auctions[:, self.column_map["current_hours"]]
        time_left     = auctions[:, self.column_map["time_left"]]
        y = hours_on_sale / 48.0

        return (auctions, item_index_tensor, contexts, bonus_lists,
                modifier_types, modifier_values, current_hours,
                time_left, buyout_rank), y
