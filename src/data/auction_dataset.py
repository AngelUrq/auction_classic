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
        
        # Calculate hour of week (0-167: Mon 0am = 0, Mon 1am = 1, ..., Sun 11pm = 167)
        hour_of_week = ts.weekday() * 24 + ts.hour  # weekday(): Mon=0, Sun=6

        auctions = torch.tensor(g["auctions"][:], dtype=torch.float32)
        contexts = torch.tensor(g["contexts"][:], dtype=torch.int32)
        bonus_lists = torch.tensor(g["bonus_lists"][:], dtype=torch.int32)
        modifier_types = torch.tensor(g["modifier_types"][:], dtype=torch.int32)
        modifier_values = torch.tensor(g["modifier_values"][:], dtype=torch.float32)
        buyout_rank = torch.tensor(g["buyout_ranking"][:], dtype=torch.int32)

        hours_on_sale = auctions[:, -1]
        auctions = auctions[:, :-1]

        auctions[:, self.column_map["bid"]] = torch.log1p(auctions[:, self.column_map["bid"]])
        auctions[:, self.column_map["buyout"]] = torch.log1p(auctions[:, self.column_map["buyout"]])
        modifier_values = torch.log1p(modifier_values)

        current_hours_raw = auctions[:, self.column_map["current_hours"]].clone()
        time_left_raw = auctions[:, self.column_map["time_left"]].clone()

        if self.feature_stats is not None:
            auctions = (auctions - self.feature_stats["means"][:5]) / (self.feature_stats["stds"][:5] + 1e-6)
            modifier_values = (modifier_values - self.feature_stats["modifiers_mean"]) / (self.feature_stats["modifiers_std"] + 1e-6)

        item_index_tensor = torch.tensor(item_idx, dtype=torch.int32).repeat(auctions.size(0))
        hour_of_week_tensor = torch.tensor(hour_of_week, dtype=torch.int32).repeat(auctions.size(0))
        y = hours_on_sale

        return {
            'auctions': auctions,
            'item_index': item_index_tensor,
            'contexts': contexts,
            'bonus_lists': bonus_lists,
            'modifier_types': modifier_types,
            'modifier_values': modifier_values,
            'hour_of_week': hour_of_week_tensor,
            'current_hours_raw': current_hours_raw,
            'time_left_raw': time_left_raw,
            'buyout_rank': buyout_rank,
            'target': y
        }
