import torch
import h5py
import numpy as np
from datetime import datetime, timedelta

class AuctionDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, idx_map, feature_stats=None, path="../generated/sequences.h5", max_hours_back=0):
        self.pairs = pairs
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
        self._idx_map = idx_map
        self.h5_path = path
        self.h5_file = None  # lazy-open per worker

    def __getstate__(self):
        # Ensure file handle isn't pickled to workers; each worker will open its own handle
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def _open_h5(self):
        if self.h5_file is None:
            # Open once per worker; bounded raw chunk cache to keep RAM stable
            self.h5_file = h5py.File(
                self.h5_path, "r",
                rdcc_nbytes=32 * 1024 * 1024,  # 32MB cache; tune if needed
                rdcc_nslots=100_000,
                rdcc_w0=0.75,
            )

    def __len__(self):
        return len(self.pairs)

    def __del__(self):
        try:
            if self.h5_file is not None:
                self.h5_file.close()
        except Exception:
            pass

    def _hour_key(self, ts):
        # Matches indices.csv/parquet formatting
        return ts.strftime("%Y-%m-%d %H:00:00")

    def __getitem__(self, idx):
        self._open_h5()

        pair = self.pairs.iloc[idx]
        ts = datetime.strptime(pair["record"], "%Y-%m-%d %H:%M:%S")
        item_idx = int(pair["item_index"])

        # Require current hour to exist (mirror your previous behavior)
        current_key = self._hour_key(ts)
        if (item_idx, current_key) not in self._idx_map:
            raise KeyError(f"Current timestamp data missing for {current_key}:{item_idx}")

        # Collect the window [ts - max_hours_back, ..., ts]
        # For each hour that exists, record (start, length, hour_of_week, offset)
        present = []
        for hours_back in range(self.max_hours_back, -1, -1):
            t = ts - timedelta(hours=hours_back)
            k = self._hour_key(t)
            key = (item_idx, k)
            if key in self._idx_map:
                start, length = self._idx_map[key]
                if length > 0:
                    hour_of_week = t.weekday() * 24 + t.hour
                    present.append((start, length, hour_of_week, hours_back))

        # Sort by start to match on-disk chronological layout, then compute one contiguous read
        present.sort(key=lambda x: x[0])  # by start
        earliest_start = present[0][0]
        latest_end = present[-1][0] + present[-1][1]

        grp = self.h5_file[f"items/{item_idx}"]

        # Single contiguous read per dataset
        auctions_np = grp["data"][earliest_start:latest_end]                   # (N, 6) float32
        contexts_np = grp["contexts"][earliest_start:latest_end]               # (N,) int32
        bonus_lists_np = grp["bonus_lists"][earliest_start:latest_end]         # (N, 9) int32
        modifier_types_np = grp["modifier_types"][earliest_start:latest_end]   # (N, 11) int32
        modifier_values_np = grp["modifier_values"][earliest_start:latest_end] # (N, 11) float32

        # Build hour_of_week and time_offset expanded per row using the per-hour lengths
        hour_of_week_vals = []
        time_offset_vals = []
        for start, length, how, off in present:
            hour_of_week_vals.append(np.full(length, how, dtype=np.int32))
            time_offset_vals.append(np.full(length, off, dtype=np.int32))

        hour_of_week_np = np.concatenate(hour_of_week_vals, axis=0) if hour_of_week_vals else np.empty((0,), dtype=np.int32)
        time_offset_np = np.concatenate(time_offset_vals, axis=0) if time_offset_vals else np.empty((0,), dtype=np.int32)

        # Convert to torch
        auctions = torch.tensor(auctions_np, dtype=torch.float32)
        contexts = torch.tensor(contexts_np, dtype=torch.int32).long()
        bonus_lists = torch.tensor(bonus_lists_np, dtype=torch.int32).long()
        modifier_types = torch.tensor(modifier_types_np, dtype=torch.int32).long()
        modifier_values = torch.tensor(modifier_values_np, dtype=torch.float32)

        hour_of_week = torch.tensor(hour_of_week_np, dtype=torch.int32)
        time_offset = torch.tensor(time_offset_np, dtype=torch.int32)

        # Target = hours_on_sale (last column), then drop it
        hours_on_sale = auctions[:, self.column_map["hours_on_sale"]]
        auctions = auctions[:, :-1]

        # Log transforms
        auctions[:, self.column_map["bid"]] = torch.log1p(auctions[:, self.column_map["bid"]])
        auctions[:, self.column_map["buyout"]] = torch.log1p(auctions[:, self.column_map["buyout"]])
        modifier_values = torch.log1p(modifier_values)

        # Preserve raw fields
        current_hours_raw = auctions[:, self.column_map["current_hours"]].clone()
        time_left_raw = auctions[:, self.column_map["time_left"]].clone()

        # Optional standardization
        if self.feature_stats is not None:
            auctions = (auctions - self.feature_stats["means"][:5]) / (self.feature_stats["stds"][:5] + 1e-6)
            modifier_values = (modifier_values - self.feature_stats["modifiers_mean"]) / (self.feature_stats["modifiers_std"] + 1e-6)

        # item_index tensor per row
        item_index_tensor = torch.tensor(item_idx, dtype=torch.int32).repeat(auctions.size(0))

        if auctions.shape[0] != hour_of_week.shape[0]:
            print(f'auctions.shape: {auctions.shape}')
            print(f'hour_of_week.shape: {hour_of_week.shape}')
            print(f'item index: {item_idx}')
            print(f'timestamp: {ts}')
            print(f'present: {present}')
            print('problem here')
            print()
            print('-------------------------------------------------------------------')

        y = hours_on_sale

        return {
            'auctions': auctions,
            'item_index': item_index_tensor,
            'contexts': contexts,
            'bonus_lists': bonus_lists,
            'modifier_types': modifier_types,
            'modifier_values': modifier_values,
            'hour_of_week': hour_of_week,
            'time_offset': time_offset,
            'current_hours_raw': current_hours_raw,
            'time_left_raw': time_left_raw,
            'target': y
        }
