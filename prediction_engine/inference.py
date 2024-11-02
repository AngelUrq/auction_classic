import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

@torch.no_grad()
def predict_dataframe(model, df_auctions, prediction_time,  time_left_mapping, item_to_index, lambda_value=0.0401, device='cpu'):
    model.eval()

    auctions_by_item = {}

    for group_item_id in tqdm(df_auctions['item_id'].unique()):

        auctions_by_item[group_item_id] = []
        df_auctions_group = df_auctions[df_auctions['item_id'] == group_item_id]

        for index, auction in df_auctions_group.iterrows():
            auction_id = auction['id']
            item_id = auction['item_id']
            time_left_numeric = time_left_mapping[auction['time_left']] / 48.0
            bid = np.log1p(auction['bid']) / 15.0
            buyout = np.log1p(auction['buyout']) / 15.0
            quantity = auction['quantity'] / 200.0
            item_index = item_to_index.get(item_id, 1)
            hours_since_first_appearance = auction['hours_since_first_appearance'] / 48.0

            processed_auction = [
                auction_id,
                bid, 
                buyout,  
                quantity, 
                item_index,
                time_left_numeric, 
                hours_since_first_appearance
            ]
            
            auctions_by_item[group_item_id].append(processed_auction)

    df_auctions['prediction'] = 0.0
    df_auctions['sale_probability'] = 0.0

    for group_item_id in tqdm(list(auctions_by_item.keys())):
        auction_data = np.array(auctions_by_item[group_item_id])

        X = torch.tensor(auction_data[:, 1:], dtype=torch.float32).to(device)
        lengths = torch.tensor([len(auction_data)])
        y = model(X.unsqueeze(0), lengths)

        for i, auction_id in enumerate(auction_data[:, 0]):
            df_auctions.loc[df_auctions['id'] == auction_id, 'prediction'] = round(y[0][i].item(), 2)
            df_auctions.loc[df_auctions['id'] == auction_id, 'sale_probability'] = np.exp(-lambda_value * y[0][i].item())

    return df_auctions
