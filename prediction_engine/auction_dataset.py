import torch
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class AuctionDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, item_to_index, path='../generated/sequences.h5', weekly_hours='../generated/weekly_hours.csv'):
        self.pairs = pairs
        self.column_map = {
            'item_id': 0,
            'bid': 1,
            'buyout': 2,
            'quantity': 3,
            'time_left': 4,
            'hours_since_first_appearance': 5,
            'hour': 6,
            'weekday': 7,
            'average_hours_on_sale': 8,
        }
        self.item_to_index = item_to_index
        self.path = path
        
        weekly_hours = pd.read_csv(weekly_hours)
        self.weekly_hours = {}
        
        # Convert dates to datetime objects for easier comparison
        for _, row in weekly_hours.iterrows():
            item_id = row['item_id']
            date = datetime.strptime(row['date'], "%Y-%m-%d").date()
            hours = row['hours_on_sale']
            
            if item_id not in self.weekly_hours:
                self.weekly_hours[item_id] = {}
            
            self.weekly_hours[item_id][date] = hours
        
        print(f"Dataset size: {len(self)}")
    
    def find_nearest_date_hours(self, item_id, target_date):
        if item_id not in self.weekly_hours:
            return 0
        
        item_dates = list(self.weekly_hours[item_id].keys())
        if not item_dates:
            return 0
        
        valid_dates = [date for date in item_dates if date <= target_date]
        
        if not valid_dates:
            return 0
        
        closest_date = max(valid_dates)
        return self.weekly_hours[item_id][closest_date]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        
        record = pair['record']
        item_id = pair['item_id']
        
        date_time_obj = datetime.strptime(record, "%Y-%m-%d %H:%M:%S")
        date_folder_name = date_time_obj.strftime("%Y-%m-%d")
        hour_folder_name = date_time_obj.strftime("%H")
        
        hour = date_time_obj.hour
        weekday = date_time_obj.weekday()
        
        with h5py.File(self.path, 'r') as f:
            data = f[f'{date_folder_name}/{hour_folder_name}/{item_id}'][:]
        
        X = torch.tensor(data, dtype=torch.float32)
        
        y = X[:, -1]
        X = X[:, :-1]
        
        target_date = date_time_obj.date()
        average_hours_on_sale = self.weekly_hours.get(item_id, {}).get(target_date, None)
        
        if average_hours_on_sale is None:
            average_hours_on_sale = self.find_nearest_date_hours(item_id, target_date)
        
        # Add columns for hour, weekday, and average_hours_on_sale
        X = torch.cat((X, torch.zeros((X.size(0), 3))), dim=1)
        
        X[:, self.column_map['item_id']] = torch.tensor(
            [self.item_to_index.get(int(item), 1) for item in X[:, self.column_map['item_id']]], 
            dtype=torch.long
        )
        X[:, self.column_map['time_left']] = X[:, self.column_map['time_left']] / 48.0
        X[:, self.column_map['hours_since_first_appearance']] = X[:, self.column_map['hours_since_first_appearance']] / 48.0
        X[:, self.column_map['bid']] = torch.log1p(X[:, self.column_map['bid']]) / 15.0
        X[:, self.column_map['buyout']] = torch.log1p(X[:, self.column_map['buyout']]) / 15.0
        X[:, self.column_map['quantity']] = X[:, self.column_map['quantity']] / 200.0
        X[:, self.column_map['hour']] = np.sin(2 * np.pi * hour / 24)
        X[:, self.column_map['weekday']] = np.sin(2 * np.pi * weekday / 7)
        X[:, self.column_map['average_hours_on_sale']] = average_hours_on_sale / 48.0
        
        # Randomly permute the sequences
        indices = torch.randperm(X.size(0))
        X = X[indices]
        y = y[indices]
        
        return X, y
        