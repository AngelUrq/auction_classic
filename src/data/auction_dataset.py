import os
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset


class AuctionDataset(Dataset):
    def __init__(
        self,
        pairs,
        idx_map_global,
        feature_stats=None,
        root="generated/memmap",
        max_hours_back=0,
    ):
        """
        Memmap-backed dataset using global .npy arrays.

        root is the directory created by convert_hdf5_to_npy.py and must contain:
            data.npy
            contexts.npy
            bonus_ids.npy
            modifier_types.npy
            modifier_values.npy
            idx_map_global.pkl
        """
        self.pairs = pairs.reset_index(drop=True)
        self.idx_map = idx_map_global
        self.feature_stats = feature_stats
        self.max_hours_back = int(max_hours_back)

        self.column_map = {
            "bid": 0,
            "buyout": 1,
            "quantity": 2,
            "time_left": 3,
            "listing_age": 4,
            "listing_duration": 5,
        }

        data_mm = np.memmap(os.path.join(root, "data.npy"), mode="r", dtype=np.float32)
        total_rows = data_mm.size // 6
        self.data = data_mm.reshape((total_rows, 6))

        self.contexts = np.memmap(
            os.path.join(root, "contexts.npy"), mode="r", dtype=np.int32
        )

        self.bonus_ids = np.memmap(
            os.path.join(root, "bonus_ids.npy"), mode="r", dtype=np.int32
        ).reshape((total_rows, 9))

        self.modifier_types = np.memmap(
            os.path.join(root, "modifier_types.npy"), mode="r", dtype=np.int32
        ).reshape((total_rows, 11))

        self.modifier_values = np.memmap(
            os.path.join(root, "modifier_values.npy"), mode="r", dtype=np.float32
        ).reshape((total_rows, 11))

    def __len__(self):
        return len(self.pairs)

    def _hour_key(self, ts: datetime) -> str:
        return ts.strftime("%Y-%m-%d %H:00:00")

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        ts = datetime.strptime(pair["record"], "%Y-%m-%d %H:%M:%S")
        item_idx = int(pair["item_index"])

        current_key = self._hour_key(ts)
        if (item_idx, current_key) not in self.idx_map:
            raise KeyError(f"Missing timestamp for {current_key}:{item_idx}")

        present = []
        for hours_back in range(self.max_hours_back, -1, -1):
            t = ts - timedelta(hours=hours_back)
            k = self._hour_key(t)
            key = (item_idx, k)
            if key in self.idx_map:
                start, length = self.idx_map[key]
                if length > 0:
                    hour_of_week = t.weekday() * 24 + t.hour
                    present.append((start, length, hour_of_week, hours_back))

        if not present:
            raise RuntimeError(f"No sequences for item_idx={item_idx}, ts={ts}")

        present.sort(key=lambda x: x[0])
        earliest_start = present[0][0]
        latest_end = present[-1][0] + present[-1][1]

        window = slice(earliest_start, latest_end)

        auction_features_np = self.data[window]
        contexts_np = self.contexts[window]
        bonus_ids_np = self.bonus_ids[window]
        modifier_types_np = self.modifier_types[window]
        modifier_values_np = self.modifier_values[window]

        hour_of_week_vals = []
        snapshot_offset_vals = []
        for start, length, how, off in present:
            hour_of_week_vals.append(np.full(length, how, dtype=np.int32))
            snapshot_offset_vals.append(np.full(length, off, dtype=np.int32))

        hour_of_week_np = np.concatenate(hour_of_week_vals)
        snapshot_offset_np = np.concatenate(snapshot_offset_vals)

        auction_features = torch.tensor(auction_features_np, dtype=torch.float32)
        modifier_values = torch.tensor(modifier_values_np, dtype=torch.float32)

        contexts = torch.tensor(contexts_np, dtype=torch.int32)
        bonus_ids = torch.tensor(bonus_ids_np, dtype=torch.int32)
        modifier_types = torch.tensor(modifier_types_np, dtype=torch.int32)

        hour_of_week = torch.tensor(hour_of_week_np, dtype=torch.int32)
        snapshot_offset = torch.tensor(snapshot_offset_np, dtype=torch.int32)

        listing_duration = auction_features[:, self.column_map["listing_duration"]]
        auction_features = auction_features[:, :-1]

        auction_features[:, self.column_map["bid"]] = torch.log1p(
            auction_features[:, self.column_map["bid"]]
        )
        auction_features[:, self.column_map["buyout"]] = torch.log1p(
            auction_features[:, self.column_map["buyout"]]
        )
        modifier_values = torch.log1p(modifier_values)

        listing_age = auction_features[:, self.column_map["listing_age"]].clone()
        time_left = auction_features[:, self.column_map["time_left"]].clone()

        if self.feature_stats is not None:
            means = self.feature_stats["means"][:5].float()
            stds = self.feature_stats["stds"][:5].float()
            auction_features = (auction_features - means) / (stds + 1e-6)

            mod_mean = self.feature_stats["modifiers_mean"].float()
            mod_std = self.feature_stats["modifiers_std"].float()
            modifier_values = (modifier_values - mod_mean) / (mod_std + 1e-6)

        item_index_tensor = torch.full(
            (auction_features.size(0),), item_idx, dtype=torch.int32
        )

        y = listing_duration

        return {
            "auction_features": auction_features,
            "item_index": item_index_tensor,
            "contexts": contexts,
            "bonus_ids": bonus_ids,
            "modifier_types": modifier_types,
            "modifier_values": modifier_values,
            "hour_of_week": hour_of_week,
            "snapshot_offset": snapshot_offset,
            "listing_age": listing_age,
            "time_left": time_left,
            "listing_duration": y,
        }
