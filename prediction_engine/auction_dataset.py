import torch
from datetime import datetime

class AuctionDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, item_to_index, weekly_historical_prices, path='sequences'):
        self.pairs = pairs
        self.column_map = {
            'bid': 0,
            'buyout': 1,
            'quantity': 2,
            'item_id': 3,
            'time_left': 4,
            'hours_since_first_appearance': 5,
            'historical_price': 6
        }
        self.item_to_index = item_to_index
        self.path = path
        self.weekly_historical_prices = weekly_historical_prices.copy()
        self.weekly_historical_prices['datetime'] = self.weekly_historical_prices['datetime'].astype(str)
        self.weekly_historical_prices.set_index(['item_id', 'datetime'], inplace=True)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        record = pair['record']
        item_id = pair['item_id']
        date_time_obj = datetime.strptime(record, "%Y-%m-%d %H:%M:%S")
        date_folder_name = date_time_obj.strftime("%d-%m-%Y")
        hour_folder_name = date_time_obj.strftime("%H")
        datetime_str = date_time_obj.strftime("%Y-%m-%d %H:%M:%S").split(' ')[0] + ' 00:00:00'
        data = torch.load(f'{self.path}/{date_folder_name}/{hour_folder_name}.pt')
        X = data[str(item_id)]
        y = X[:, -1]
        X = X[:, :-1]

        if (item_id, datetime_str) in self.weekly_historical_prices.index:
            historical_price = self.weekly_historical_prices.loc[item_id, datetime_str]['price']
        else:
            historical_price = X[:, self.column_map['buyout']].median()

        X = torch.cat([X, torch.ones(X.shape[0], 1) * historical_price], dim=1)
        X[:, self.column_map['bid']] = X[:, self.column_map['bid']] * 10000 / 1000
        X[:, self.column_map['buyout']] = X[:, self.column_map['buyout']] * 10000 / 1000
        X[:, self.column_map['quantity']] = X[:, self.column_map['quantity']] / 200.0
        return X, y
