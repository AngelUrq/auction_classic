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
    buyout_ranking = [item['buyout_rank'] for item in batch]
    hour_of_week = [item['hour_of_week'] for item in batch]
    targets = [item['target'] for item in batch]

    # Pad all tensors to max size
    auctions = pad_tensors_to_max_size(auctions)
    item_index = pad_tensors_to_max_size(item_index)
    contexts = pad_tensors_to_max_size(contexts)
    bonus_lists = pad_tensors_to_max_size(bonus_lists)
    modifier_types = pad_tensors_to_max_size(modifier_types)
    modifier_values = pad_tensors_to_max_size(modifier_values)
    buyout_ranking = pad_tensors_to_max_size(buyout_ranking)
    current_hours = pad_tensors_to_max_size(current_hours)
    time_left = pad_tensors_to_max_size(time_left)
    hour_of_week = pad_tensors_to_max_size(hour_of_week)
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
        'buyout_rank': buyout_ranking,
        'hour_of_week': hour_of_week,
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

def load_auctions_from_sample(data_dir, prediction_time, time_left_mapping, item_to_idx, context_to_idx, bonus_to_idx, modtype_to_idx, last_days=None):
    file_info = {}
    auction_appearances = {}

    for root, dirs, files in os.walk(data_dir):
        for filename in tqdm(files):
            filepath = os.path.join(root, filename)
            date = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')

            if last_days and prediction_time is not None:
                if date < prediction_time - timedelta(days=last_days):
                    continue
                    
            file_info[filepath] = date

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}
    
    raw_auctions = []
    
    for filepath in list(file_info.keys()):
        print(filepath)
        with open(filepath, 'r') as f:
            try:
                json_data = json.load(f)
                
                if 'auctions' not in json_data:
                    print(f"File {filepath} does not contain 'auctions' key, skipping.")
                    continue
                
                auction_data = json_data['auctions']
                timestamp = file_info[filepath]
                
                for auction in auction_data:
                    auction_id = auction['id']

                    if auction_id not in auction_appearances:
                        auction_appearances[auction_id] = {'first': timestamp, 'last': timestamp}
                    else:
                        auction_appearances[auction_id]['last'] = timestamp
                
                if prediction_time is not None:
                    if timestamp == prediction_time:
                        raw_auctions.extend(auction_data)
                else:
                    raw_auctions.extend(auction_data)

            except json.JSONDecodeError as e:
                print(f"Error loading file {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error loading file {filepath}: {e}")
                continue

    auctions = []
    for auction in tqdm(raw_auctions):
        try: 
            first_appearance = auction_appearances[auction['id']]['first']
            last_appearance = auction_appearances[auction['id']]['last']

            auction_id = auction['id']
            item_id = auction['item']['id']
            item_index = item_to_idx.get(str(item_id), 1)
            bid = auction.get('bid', 0) / 10000.0
            buyout = auction['buyout'] / 10000.0
            quantity = auction['quantity']
            time_left = time_left_mapping[auction['time_left']]
            context = context_to_idx[str(auction['item'].get('context', 0))]
            bonus_lists = [bonus_to_idx.get(str(bonus), 1) for bonus in auction['item'].get('bonus_lists', [])]
            modifiers = auction['item'].get('modifiers', [])

            modifier_types = []
            modifier_values = []

            for modifier in modifiers:
                modifier_types.append(modtype_to_idx[str(modifier['type'])])
                modifier_values.append(modifier['value'])

            if 'pet_species_id' in auction['item']:
                continue

            first_appearance = first_appearance.strftime('%Y-%m-%d %H:%M:%S')
            last_appearance = last_appearance.strftime('%Y-%m-%d %H:%M:%S')

            auctions.append([
                auction_id,
                item_index,
                bid,
                buyout,
                quantity,
                time_left,
                context,
                bonus_lists,
                    modifier_types,
                    modifier_values,
                    first_appearance,
                    last_appearance
                ])
        except Exception as e:
            print(f"Unexpected error processing auction {auction['id']}: {e}")
            continue
        
    df_auctions = pd.DataFrame(auctions, columns=['id', 'item_index', 'bid', 'buyout', 'quantity', 'time_left', 'context', 'bonus_lists', 'modifier_types', 'modifier_values', 'first_appearance', 'last_appearance'])
    df_auctions['first_appearance'] = pd.to_datetime(df_auctions['first_appearance'])
    df_auctions['last_appearance'] = pd.to_datetime(df_auctions['last_appearance'])

    df_auctions = df_auctions[(df_auctions['first_appearance'] <= prediction_time) & (df_auctions['last_appearance'] >= prediction_time)]

    print(f'Processing {len(df_auctions)} auctions')

    df_auctions['current_hours'] = (prediction_time - df_auctions['first_appearance']).dt.total_seconds() / 3600
    df_auctions['hours_on_sale'] = (df_auctions['last_appearance'] - prediction_time).dt.total_seconds() / 3600

    return df_auctions
