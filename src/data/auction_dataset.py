import torch
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

class AuctionDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_stats=None, path='../generated/sequences.h5'):
        self.pairs = pairs
        self.feature_stats = feature_stats
        self.column_map = {
            'bid': 0,
            'buyout': 1,
            'quantity': 2,
            'time_left': 3,
            'current_hours': 4,
            'hour': 5,
            'weekday': 6,
            'hours_on_sale': 7
        }

        self.path = path
        self.h5_file = h5py.File(path, 'r')
        
        print(f"Dataset size: {len(self)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __del__(self):
        self.h5_file.close()

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        
        record = pair['record']
        item_index = pair['item_index']
        
        date_time_obj = datetime.strptime(record, "%Y-%m-%d %H:%M:%S")
        date_folder_name = date_time_obj.strftime("%Y-%m-%d")
        hour_folder_name = date_time_obj.strftime("%H")
        
        hour = date_time_obj.hour
        weekday = date_time_obj.weekday()
        
        auctions = self.h5_file[f'{date_folder_name}/{hour_folder_name}/{item_index}/auctions'][:]
        contexts = self.h5_file[f'{date_folder_name}/{hour_folder_name}/{item_index}/contexts'][:]
        bonus_lists = self.h5_file[f'{date_folder_name}/{hour_folder_name}/{item_index}/bonus_lists'][:]
        modifier_types = self.h5_file[f'{date_folder_name}/{hour_folder_name}/{item_index}/modifier_types'][:]
        modifier_values = self.h5_file[f'{date_folder_name}/{hour_folder_name}/{item_index}/modifier_values'][:]
    
        auctions = torch.tensor(auctions, dtype=torch.float32)
        item_index = torch.tensor(item_index, dtype=torch.int32).repeat(auctions.shape[0])
        contexts = torch.tensor(contexts, dtype=torch.int32)
        bonus_lists = torch.tensor(bonus_lists, dtype=torch.int32)
        modifier_types = torch.tensor(modifier_types, dtype=torch.int32)
        modifier_values = torch.tensor(modifier_values, dtype=torch.float32)

        hours_on_sale = auctions[:, -1]
        auctions = auctions[:, :-1]
        
        # Add columns for hour, weekday
        auctions = torch.cat((auctions, torch.zeros((auctions.size(0), 2))), dim=1)

        # Apply log1p transformation before normalization for bid and buyout
        auctions[:, self.column_map['bid']] = torch.log1p(auctions[:, self.column_map['bid']])
        auctions[:, self.column_map['buyout']] = torch.log1p(auctions[:, self.column_map['buyout']])

        # Apply log1p transformation before normalization for modifier_values
        modifier_values = torch.log1p(modifier_values)
        
        # Add hour and weekday features
        auctions[:, self.column_map['hour']] = np.sin(2 * np.pi * hour / 24)
        auctions[:, self.column_map['weekday']] = np.sin(2 * np.pi * weekday / 7)

        current_hours = auctions[:, self.column_map['current_hours']]

        if self.feature_stats:
            # Normalize auction features
            auctions_mean = self.feature_stats['means']
            auctions_std = self.feature_stats['stds']

            auctions = (auctions - auctions_mean) / (auctions_std + 1e-6)

            # Normalize modifier values
            modifiers_mean = self.feature_stats['modifiers_mean']
            modifiers_std = self.feature_stats['modifiers_std']

            modifier_values = (modifier_values - modifiers_mean) / (modifiers_std + 1e-6)
            
        y = hours_on_sale / 48.0
        
        return (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values, current_hours), y
        