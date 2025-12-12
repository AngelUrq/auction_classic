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
            bonus_lists.npy
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
            "current_hours": 4,
            "hours_on_sale": 5,
        }

        data_mm = np.memmap(os.path.join(root, "data.npy"), mode="r", dtype=np.float32)
        total_rows = data_mm.size // 6
        self.data = data_mm.reshape((total_rows, 6))

        self.contexts = np.memmap(
            os.path.join(root, "contexts.npy"), mode="r", dtype=np.int32
        )

        self.bonus_lists = np.memmap(
            os.path.join(root, "bonus_lists.npy"), mode="r", dtype=np.int32
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

        auctions_np = self.data[window]
        contexts_np = self.contexts[window]
        bonus_lists_np = self.bonus_lists[window]
        modifier_types_np = self.modifier_types[window]
        modifier_values_np = self.modifier_values[window]

        hour_of_week_vals = []
        time_offset_vals = []
        for start, length, how, off in present:
            hour_of_week_vals.append(np.full(length, how, dtype=np.int32))
            time_offset_vals.append(np.full(length, off, dtype=np.int32))

        hour_of_week_np = np.concatenate(hour_of_week_vals)
        time_offset_np = np.concatenate(time_offset_vals)

        auctions = torch.tensor(auctions_np, dtype=torch.float32)
        modifier_values = torch.tensor(modifier_values_np, dtype=torch.float32)

        contexts = torch.tensor(contexts_np, dtype=torch.int32)
        bonus_lists = torch.tensor(bonus_lists_np, dtype=torch.int32)
        modifier_types = torch.tensor(modifier_types_np, dtype=torch.int32)

        hour_of_week = torch.tensor(hour_of_week_np, dtype=torch.int32)
        time_offset = torch.tensor(time_offset_np, dtype=torch.int32)

        hours_on_sale = auctions[:, self.column_map["hours_on_sale"]]
        auctions = auctions[:, :-1]

        auctions[:, self.column_map["bid"]] = torch.log1p(
            auctions[:, self.column_map["bid"]]
        )
        auctions[:, self.column_map["buyout"]] = torch.log1p(
            auctions[:, self.column_map["buyout"]]
        )
        modifier_values = torch.log1p(modifier_values)

        current_hours_raw = auctions[:, self.column_map["current_hours"]].clone()
        time_left_raw = auctions[:, self.column_map["time_left"]].clone()

        if self.feature_stats is not None:
            means = self.feature_stats["means"][:5].float()
            stds = self.feature_stats["stds"][:5].float()
            auctions = (auctions - means) / (stds + 1e-6)
            
            mod_mean = self.feature_stats["modifiers_mean"].float()
            mod_std = self.feature_stats["modifiers_std"].float()
            modifier_values = (modifier_values - mod_mean) / (mod_std + 1e-6)

        item_index_tensor = torch.full(
            (auctions.size(0),), item_idx, dtype=torch.int32
        )

        y = hours_on_sale

        return {
            "auctions": auctions,
            "item_index": item_index_tensor,
            "contexts": contexts,
            "bonus_lists": bonus_lists,
            "modifier_types": modifier_types,
            "modifier_values": modifier_values,
            "hour_of_week": hour_of_week,
            "time_offset": time_offset,
            "current_hours_raw": current_hours_raw,
            "time_left_raw": time_left_raw,
            "target": y,
        }
