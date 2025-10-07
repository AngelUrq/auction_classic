import torch
import torch.nn.functional as F
import json
import requests
import time
import os
import math
import pandas as pd
from tqdm import tqdm
from datetime import timedelta, datetime

CLIENT_KEY = "c39078bd5f0f4e798a3a1b734dd9d280"
SECRET_KEY = "4UEplhA8jYa8wvX58C5QdV7JDTDY9rNX"
REALM_ID = "3676"

def pad_tensors_to_max_size(tensor_list):
    # Find maximum dimensions
    max_dims = []
    for dim in range(tensor_list[0].dim()):
        max_dim_size = max([tensor.size(dim) for tensor in tensor_list])
        max_dims.append(max_dim_size)
    
    # Pad each tensor to match max dimensions
    padded_tensors = []
    for tensor in tensor_list:
        # Calculate padding for each dimension
        pad_sizes = []
        for dim in range(tensor.dim()):
            pad_size = max_dims[dim] - tensor.size(dim)
            # Padding is applied from the last dimension backward
            # For each dimension, we need (padding_left, padding_right)
            # We'll add all padding to the right side
            pad_sizes = [0, pad_size] + pad_sizes
        
        # Apply padding
        padded_tensor = F.pad(tensor, pad_sizes, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    
    return torch.stack(padded_tensors)

def collate_auctions(batch):
    # Extract lists of each field from the batch of dictionaries
    auctions = [item['auctions'] for item in batch]
    item_index = [item['item_index'] for item in batch]
    contexts = [item['contexts'] for item in batch]
    bonus_lists = [item['bonus_lists'] for item in batch]
    modifier_types = [item['modifier_types'] for item in batch]
    modifier_values = [item['modifier_values'] for item in batch]
    current_hours = [item['current_hours_raw'] for item in batch]
    time_left = [item['time_left_raw'] for item in batch]
    hour_of_week = [item['hour_of_week'] for item in batch]
    time_offset = [item['time_offset'] for item in batch]
    targets = [item['target'] for item in batch]

    # Pad all tensors to max size
    auctions = pad_tensors_to_max_size(auctions)
    item_index = pad_tensors_to_max_size(item_index)
    contexts = pad_tensors_to_max_size(contexts)
    bonus_lists = pad_tensors_to_max_size(bonus_lists)
    modifier_types = pad_tensors_to_max_size(modifier_types)
    modifier_values = pad_tensors_to_max_size(modifier_values)
    current_hours = pad_tensors_to_max_size(current_hours)
    time_left = pad_tensors_to_max_size(time_left)
    hour_of_week = pad_tensors_to_max_size(hour_of_week)
    time_offset = pad_tensors_to_max_size(time_offset)
    targets = pad_tensors_to_max_size(targets)

    # Return as dictionary for consistency
    return {
        'auctions': auctions,
        'item_index': item_index,
        'contexts': contexts,
        'bonus_lists': bonus_lists,
        'modifier_types': modifier_types,
        'modifier_values': modifier_values,
        'current_hours_raw': current_hours,
        'time_left_raw': time_left,
        'hour_of_week': hour_of_week,
        'time_offset': time_offset,
        'target': targets
    }

def create_access_token(client_id, client_secret, region='us'):
    """Create an OAuth access token for the Blizzard API.
    
    This matches the approach used in the shell script, which uses a direct POST request
    with client credentials authentication.
    """
    data = {'grant_type': 'client_credentials'}
    response = requests.post(
        f'https://{region}.battle.net/oauth/token',
        data=data,
        auth=(client_id, client_secret)
    )
    
    if response.status_code != 200:
        print(f"Error: Failed to obtain access token. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return {'access_token': None}
        
    return response.json()

def get_current_auctions(config):
    response = create_access_token(CLIENT_KEY, SECRET_KEY)
    token = response['access_token']
    
    if not token:
        print("Error: Failed to obtain access token")
        return []
        
    print('Token created')
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    
    url = (
        'https://us.api.blizzard.com/data/wow/connected-realm/'
        f"{REALM_ID}/auctions"
        '?namespace=dynamic-us&locale=en_US'
    )
    
    response = requests.get(url, headers=headers)
    print('Request done')
    
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return []
    
    data = response.json()
    
    return data

def load_auctions_from_sample(
    data_dir,
    prediction_time,
    time_left_mapping,
    item_to_idx,
    context_to_idx,
    bonus_to_idx,
    modtype_to_idx,
    max_hours_back=0,
    include_targets=True
):
    # ---- define window ----
    past_hours = 48 + max_hours_back
    future_hours = 48 + max_hours_back if include_targets else 0
    window_start = prediction_time - timedelta(hours=past_hours)
    window_end = prediction_time + timedelta(hours=future_hours)

    # ---- collect files in window (sorted) ----
    file_info = {}
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            try:
                ts = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')
            except Exception:
                continue
            if window_start <= ts <= window_end:
                file_info[os.path.join(root, filename)] = ts
    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda it: it[1])}

    # ---- PASS 1: track first/last appearance per auction ----
    auction_appearances = {}
    for filepath, snapshot_time in tqdm(list(file_info.items()), desc="Pass 1/2: appearances"):
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
        except Exception:
            continue

        for auction in json_data.get('auctions', []):
            if 'pet_species_id' in auction.get('item', {}):
                continue
            auction_id = auction.get('id')
            if auction_id is None:
                continue

            if auction_id not in auction_appearances:
                auction_appearances[auction_id] = {'first': snapshot_time, 'last': snapshot_time}
            else:
                auction_appearances[auction_id]['last'] = snapshot_time

    # ---- PASS 2: build rows while reading one file at a time ----
    rows = []
    for filepath, snapshot_time in tqdm(list(file_info.items()), desc="Pass 2/2: rows"):
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
        except Exception:
            continue

        for auction in json_data.get('auctions', []):
            if 'pet_species_id' in auction.get('item', {}):
                continue
            auction_id = auction.get('id')
            if auction_id is None:
                continue

            item_id = auction['item']['id']
            item_index = item_to_idx.get(str(item_id), 1)

            bid = auction.get('bid', 0) / 10000.0
            buyout = auction.get('buyout', 0) / 10000.0
            quantity = auction.get('quantity', 1)

            time_left_key = auction.get('time_left')
            time_left = time_left_mapping.get(time_left_key, 0.0)

            context = context_to_idx.get(str(auction['item'].get('context', 0)), 1)
            bonus_lists = [bonus_to_idx.get(str(b), 1) for b in auction['item'].get('bonus_lists', [])]

            modifier_types = []
            modifier_values = []
            for modifier in auction['item'].get('modifiers', []):
                modifier_types.append(modtype_to_idx.get(str(modifier.get('type')), 1))
                modifier_values.append(modifier.get('value', 0))

            first_appearance = auction_appearances[auction_id]['first']
            last_appearance = auction_appearances[auction_id]['last']

            current_hours = (snapshot_time - first_appearance).total_seconds() / 3600.0
            if include_targets:
                hours_on_sale = (last_appearance - snapshot_time).total_seconds() / 3600.0

            time_offset = int((prediction_time - snapshot_time).total_seconds() // 3600)  # negative if future
            hour_of_week = snapshot_time.weekday() * 24 + snapshot_time.hour  # 0..167

            if include_targets:
                rows.append([
                    auction_id, item_index, bid, buyout, quantity, time_left, context,
                    bonus_lists, modifier_types, modifier_values,
                    snapshot_time, time_offset, hour_of_week, current_hours, hours_on_sale
                ])
            else:
                rows.append([
                    auction_id, item_index, bid, buyout, quantity, time_left, context,
                    bonus_lists, modifier_types, modifier_values,
                    snapshot_time, time_offset, hour_of_week, current_hours
                ])

    if include_targets:
        df_auctions = pd.DataFrame(
            rows,
            columns=[
                'id','item_index','bid','buyout','quantity','time_left','context',
                'bonus_lists','modifier_types','modifier_values',
                'snapshot_time','time_offset','hour_of_week','current_hours','hours_on_sale'
            ]
        )
    else:
        df_auctions = pd.DataFrame(
            rows,
            columns=[
                'id','item_index','bid','buyout','quantity','time_left','context',
                'bonus_lists','modifier_types','modifier_values',
                'snapshot_time','time_offset','hour_of_week','current_hours'
            ]
        )

    if not df_auctions.empty:
        df_auctions = df_auctions.sort_values(['snapshot_time', 'item_index', 'id']).reset_index(drop=True)

    print(
        f'Built dataframe with {len(df_auctions)} rows from {len(file_info)} snapshots '
        f'[{window_start:%Y-%m-%d %H}:00, {window_end:%Y-%m-%d %H}:00], include_targets={include_targets}'
    )

    return df_auctions
