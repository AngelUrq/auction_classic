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

CLIENT_KEY = ""
SECRET_KEY = ""
REALM_ID = "3676"

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def _crop_left(t: torch.Tensor, L: int) -> torch.Tensor:
    """Keep only the last L steps (crop from the left)."""
    if t.size(0) > L:
        return t[-L:]
    return t

def _crop_and_pad(field_list, L: int | None, pad_value=0):
    """
    Crop each sequence to max L (from the left), then right-pad to batch max length.
    If L is None, do not crop—pad to the longest sequence in the batch.
    """
    if L is not None:
        cropped = [_crop_left(t, L) for t in field_list]
    else:
        cropped = field_list
        if not cropped:
            return pad_sequence([], batch_first=True, padding_value=pad_value)
        L = max(t.size(0) for t in cropped)
    return pad_sequence(cropped, batch_first=True, padding_value=pad_value)

def collate_auctions(batch, max_sequence_length=None, pad_value=0):
    """Collate function: optionally crop from the left, then pad to batch max length."""
    L = max_sequence_length

    auction_features = _crop_and_pad([b['auction_features']   for b in batch], L, pad_value)
    item_index       = _crop_and_pad([b['item_index']         for b in batch], L, pad_value)
    contexts         = _crop_and_pad([b['contexts']           for b in batch], L, pad_value)
    bonus_ids        = _crop_and_pad([b['bonus_ids']          for b in batch], L, pad_value)
    modifier_types   = _crop_and_pad([b['modifier_types']     for b in batch], L, pad_value)
    modifier_values  = _crop_and_pad([b['modifier_values']    for b in batch], L, pad_value)
    listing_age      = _crop_and_pad([b['listing_age']        for b in batch], L, pad_value)
    time_left        = _crop_and_pad([b['time_left']          for b in batch], L, pad_value)
    hour_of_week     = _crop_and_pad([b['hour_of_week']       for b in batch], L, pad_value)
    snapshot_offset  = _crop_and_pad([b['snapshot_offset']    for b in batch], L, pad_value)
    listing_duration = _crop_and_pad([b['listing_duration']   for b in batch], L, pad_value)
    is_expired       = _crop_and_pad([b['is_expired']         for b in batch], L, pad_value)
    sold             = _crop_and_pad([b['sold']               for b in batch], L, pad_value)

    return {
        'auction_features': auction_features,    # (B, T_max, 5)
        'item_index': item_index,                # (B, T_max)
        'contexts': contexts,
        'bonus_ids': bonus_ids,
        'modifier_types': modifier_types,
        'modifier_values': modifier_values,
        'listing_age': listing_age,
        'time_left': time_left,
        'hour_of_week': hour_of_week,
        'snapshot_offset': snapshot_offset,
        'listing_duration': listing_duration,
        'is_expired': is_expired,
        'sold': sold,
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
    past_hours = 48 + max_hours_back
    future_hours = 48 + max_hours_back if include_targets else 0
    window_start = prediction_time - timedelta(hours=past_hours)
    window_end = prediction_time + timedelta(hours=future_hours)

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
    skipped_items_count = 0
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
            if str(item_id) not in item_to_idx:
                skipped_items_count += 1
                continue
            item_index = item_to_idx[str(item_id)]

            bid = auction.get('bid', 0) / 10000.0
            buyout = auction.get('buyout', 0) / 10000.0
            quantity = auction.get('quantity', 1)

            time_left_key = auction.get('time_left')
            time_left = time_left_mapping.get(time_left_key, 0.0)

            context = context_to_idx.get(str(auction['item'].get('context', 0)), 1)
            bonus_ids = [bonus_to_idx.get(str(b), 1) for b in auction['item'].get('bonus_lists', [])]

            modifier_types = []
            modifier_values = []
            for modifier in auction['item'].get('modifiers', []):
                modifier_types.append(modtype_to_idx.get(str(modifier.get('type')), 1))
                modifier_values.append(modifier.get('value', 0))

            first_appearance = auction_appearances[auction_id]['first']
            last_appearance = auction_appearances[auction_id]['last']

            listing_age = (snapshot_time - first_appearance).total_seconds() / 3600.0
            if include_targets:
                listing_duration = (last_appearance - snapshot_time).total_seconds() / 3600.0

            snapshot_offset = int((prediction_time - snapshot_time).total_seconds() // 3600)  # negative if future
            hour_of_week = snapshot_time.weekday() * 24 + snapshot_time.hour  # 0..167

            if include_targets:
                rows.append([
                    auction_id, item_index, bid, buyout, quantity, time_left, context,
                    bonus_ids, modifier_types, modifier_values,
                    snapshot_time, snapshot_offset, hour_of_week, listing_age, listing_duration
                ])
            else:
                rows.append([
                    auction_id, item_index, bid, buyout, quantity, time_left, context,
                    bonus_ids, modifier_types, modifier_values,
                    snapshot_time, snapshot_offset, hour_of_week, listing_age
                ])

    if include_targets:
        df_auctions = pd.DataFrame(
            rows,
            columns=[
                'id','item_index','bid','buyout','quantity','time_left','context',
                'bonus_ids','modifier_types','modifier_values',
                'snapshot_time','snapshot_offset','hour_of_week','listing_age','listing_duration'
            ]
        )
    else:
        df_auctions = pd.DataFrame(
            rows,
            columns=[
                'id','item_index','bid','buyout','quantity','time_left','context',
                'bonus_ids','modifier_types','modifier_values',
                'snapshot_time','snapshot_offset','hour_of_week','listing_age'
            ]
        )

    if not df_auctions.empty:
        df_auctions = df_auctions.sort_values(['snapshot_time', 'item_index', 'id']).reset_index(drop=True)

    print(
        f'Built dataframe with {len(df_auctions)} rows from {len(file_info)} snapshots '
        f'[{window_start:%Y-%m-%d %H}:00, {window_end:%Y-%m-%d %H}:00], include_targets={include_targets}'
    )
    if skipped_items_count > 0:
        print(f"Skipped {skipped_items_count} items because they were not in the dictionary.")

    return df_auctions
