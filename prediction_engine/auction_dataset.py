import torch
import h5py
from datetime import datetime

class AuctionDataset(torch.utils.data.Dataset):

    def __init__(self, pairs, item_to_index, path='../generated/sequences.h5'):
        self.pairs = pairs
        self.column_map = {
            'bid': 0,
            'buyout': 1,
            'quantity': 2,
            'item_id': 3,
            'time_left': 4,
            'hours_since_first_appearance': 5
        }
        self.item_to_index = item_to_index
        self.path = path

        print(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]

        record = pair['record']
        item_id = pair['item_id']

        date_time_obj = datetime.strptime(record, "%Y-%m-%d %H:%M:%S")
        date_folder_name = date_time_obj.strftime("%Y-%m-%d")
        hour_folder_name = date_time_obj.strftime("%H")

        with h5py.File(self.path, 'r') as f:
            data = f[f'{date_folder_name}/{hour_folder_name}/{item_id}'][:]

        X = torch.tensor(data, dtype=torch.float32)

        y = X[:, -1]
        X = X[:, :-1]

        X[:, self.column_map['item_id']] = torch.tensor([self.item_to_index.get(int(item), 1) for item in X[:, self.column_map['item_id']]], dtype=torch.long)
        X[:, self.column_map['time_left']] = X[:, self.column_map['time_left']] / 48.0
        X[:, self.column_map['hours_since_first_appearance']] = X[:, self.column_map['hours_since_first_appearance']] / 48.0
        X[:, self.column_map['bid']] = torch.log1p(X[:, self.column_map['bid']]) / 15.0
        X[:, self.column_map['buyout']] = torch.log1p(X[:, self.column_map['buyout']]) / 15.0
        X[:, self.column_map['quantity']] = X[:, self.column_map['quantity']] / 200.0

        indices = torch.randperm(X.size(0))
        
        X = X[indices]
        y = y[indices]

        return X, y
        