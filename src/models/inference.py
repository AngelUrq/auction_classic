import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from src.data.utils import pad_tensors_to_max_size

@torch.no_grad()
def predict_dataframe(model, df_auctions, prediction_time, feature_stats, lambda_value=0.0401):
    model.eval()

    # Group auctions by item_index
    auctions_by_item = {}
    for group_item_index in df_auctions['item_index'].unique():
        auctions_by_item[group_item_index] = []
        df_auctions_group = df_auctions[df_auctions['item_index'] == group_item_index]

        for index, auction in df_auctions_group.iterrows():
            auction_id = auction['id']
            item_index = auction['item_index']
            time_left = auction['time_left']
            
            # Apply log1p transformation for bid and buyout
            bid = np.log1p(auction['bid'])
            buyout = np.log1p(auction['buyout'])
            quantity = auction['quantity']
            current_hours = auction['current_hours']
            
            # Get hour and weekday from prediction_time
            hour = prediction_time.hour
            weekday = prediction_time.weekday()
            
            # Convert hour and weekday to sinusoidal representation
            hour_sin = np.sin(2 * np.pi * hour / 24)
            weekday_sin = np.sin(2 * np.pi * weekday / 7)

            # Apply log1p to modifier values
            modifier_values = np.log1p(np.array(auction.get('modifier_values', [])))

            # Store the processed auction with its ID
            processed_auction = {
                'id': auction_id,
                'features': [bid, buyout, quantity, time_left, current_hours, hour_sin, weekday_sin],
                'item_index': item_index,
                'context': auction['context'],
                'bonus_lists': auction['bonus_lists'],
                'modifier_types': auction['modifier_types'],
                'modifier_values': modifier_values,
                'current_hours': current_hours
            }
            
            auctions_by_item[group_item_index].append(processed_auction)

    # Initialize prediction columns
    df_auctions['prediction'] = 0.0
    df_auctions['sale_probability'] = 0.0

    skipped_item_indices = []
    skipped_auctions = 0

    # Process each group of auctions
    for group_item_index in list(auctions_by_item.keys()):
        auctions = auctions_by_item[group_item_index]
        
        if len(auctions) > 64:
            skipped_item_indices.append(group_item_index)
            skipped_auctions += len(auctions)
            continue
            
        auction_features = torch.stack([torch.tensor(a['features'], dtype=torch.float32) for a in auctions]).to(model.device)
        auction_features = (auction_features - feature_stats['means'].to(model.device)) / (feature_stats['stds'].to(model.device) + 1e-6)
        
        item_indices = torch.tensor([a['item_index'] for a in auctions], dtype=torch.int32).to(model.device)
        contexts = torch.tensor([a['context'] for a in auctions], dtype=torch.int32).to(model.device)
        
        bonus_lists = [torch.tensor(a['bonus_lists'], dtype=torch.int32) for a in auctions]
        bonus_lists = pad_tensors_to_max_size(bonus_lists).to(model.device)
        
        modifier_types = [torch.tensor(a['modifier_types'], dtype=torch.int32) for a in auctions]
        modifier_types = pad_tensors_to_max_size(modifier_types).to(model.device)
        
        modifier_values = [torch.tensor(a['modifier_values'], dtype=torch.float32) for a in auctions]
        modifier_values = pad_tensors_to_max_size(modifier_values).to(model.device)
        
        modifier_values = (modifier_values - feature_stats['modifiers_mean'].to(model.device)) / (feature_stats['modifiers_std'].to(model.device) + 1e-6)

        # Calculate buyout ranking
        buyout_prices = auction_features[:, 1].cpu().numpy()  # Extract buyout prices
        buyout_for_ranking = np.copy(buyout_prices)
        buyout_for_ranking[buyout_for_ranking == 0] = np.inf
        
        unique_buyouts = np.unique(buyout_for_ranking[buyout_for_ranking != np.inf])
        buyout_ranking = np.zeros_like(buyout_prices, dtype=int)
        
        for i, price in enumerate(unique_buyouts, 1):
            buyout_ranking[buyout_for_ranking == price] = i
            
        buyout_ranking = torch.tensor(buyout_ranking, dtype=torch.int32).to(model.device)
    
        X = (auction_features.unsqueeze(0), item_indices.unsqueeze(0), 
             contexts.unsqueeze(0), bonus_lists.unsqueeze(0), 
             modifier_types.unsqueeze(0), modifier_values.unsqueeze(0),
             buyout_ranking.unsqueeze(0))
    
        y = model(X)
        
        for i, auction in enumerate(auctions):
            auction_id = auction['id']
            prediction_value = y[0, i, 0].item() * 48.0
            prediction_difference = prediction_value + auction['current_hours'] 

            df_auctions.loc[df_auctions['id'] == auction_id, 'prediction'] = round(prediction_value, 2)
            df_auctions.loc[df_auctions['id'] == auction_id, 'sale_probability'] = np.exp(-lambda_value * prediction_difference)

    df_auctions = df_auctions[~df_auctions['item_index'].isin(skipped_item_indices)]

    return df_auctions
