import torch
import h5py
import numpy as np
from datetime import datetime

class AuctionDataset(torch.utils.data.Dataset):
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
            "hour_cos": 6,
            "weekday_sin": 7,
            "weekday_cos": 8,
        }
        self.h5_file = h5py.File(path, "r")

    def __len__(self):
        return len(self.pairs)

    def __del__(self):
        self.h5_file.close()

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        ts = datetime.strptime(pair["record"], "%Y-%m-%d %H:%M:%S")
        item_idx = pair["item_index"]
        g = self.h5_file[f"{ts:%Y-%m-%d}/{ts:%H}/{item_idx}"]

        auctions = torch.tensor(g["auctions"][:], dtype=torch.float32)
        contexts = torch.tensor(g["contexts"][:], dtype=torch.int32)
        bonus_lists = torch.tensor(g["bonus_lists"][:], dtype=torch.int32)
        modifier_types = torch.tensor(g["modifier_types"][:], dtype=torch.int32)
        modifier_values = torch.tensor(g["modifier_values"][:], dtype=torch.float32)
        buyout_rank = torch.tensor(g["buyout_ranking"][:], dtype=torch.int32)

        hours_on_sale = auctions[:, -1]
        auctions = auctions[:, :-1]

        # Add space for hour_sin, hour_cos, weekday_sin, and weekday_cos features
        auctions = torch.cat([
            auctions,
            torch.zeros((auctions.size(0), 4))
        ], dim=1)

        hour_angle = 2 * np.pi * ts.hour / 24
        auctions[:, self.column_map["hour_sin"]] = np.sin(hour_angle)
        auctions[:, self.column_map["hour_cos"]] = np.cos(hour_angle)
        weekday_angle = 2 * np.pi * ts.weekday() / 7
        auctions[:, self.column_map["weekday_sin"]] = np.sin(weekday_angle)
        auctions[:, self.column_map["weekday_cos"]] = np.cos(weekday_angle)

        auctions[:, self.column_map["bid"]] = torch.log1p(auctions[:, self.column_map["bid"]])
        auctions[:, self.column_map["buyout"]] = torch.log1p(auctions[:, self.column_map["buyout"]])
        modifier_values = torch.log1p(modifier_values)

        current_hours_raw = auctions[:, self.column_map["current_hours"]].clone()
        time_left_raw = auctions[:, self.column_map["time_left"]].clone()

        if self.feature_stats is not None:
            auctions = (auctions - self.feature_stats["means"]) / (self.feature_stats["stds"] + 1e-6)
            modifier_values = (modifier_values - self.feature_stats["modifiers_mean"]) / (self.feature_stats["modifiers_std"] + 1e-6)

        item_index_tensor = torch.tensor(item_idx, dtype=torch.int32).repeat(auctions.size(0))
        y = hours_on_sale

        return {
            'auctions': auctions,
            'item_index': item_index_tensor,
            'contexts': contexts,
            'bonus_lists': bonus_lists,
            'modifier_types': modifier_types,
            'modifier_values': modifier_values,
            'current_hours_raw': current_hours_raw,
            'time_left_raw': time_left_raw,
            'buyout_rank': buyout_rank,
            'target': y
        }
