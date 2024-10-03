import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

data_dir = 'rl/data/auctions/'  
output_path = 'data/historical_prices.csv'  

def process_json_file(filepath):
    records = []
    try:
        date = datetime.strptime(os.path.basename(filepath)[:-5], "%Y%m%dT%H")
    except ValueError:
        print(f"Skipping file with incorrect date format: {os.path.basename(filepath)}")
        return records
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {filepath}: {e}")
        return records

    if 'auctions' in data:
        for auction in data['auctions']:
            if 'buyout' in auction and auction['buyout'] > 0:  
                buyout_in_gold = auction['buyout'] / 10000.0
                quantity = auction['quantity']
                record = {
                    'datetime': date,
                    'item_id': auction['item']['id'],
                    'price': buyout_in_gold / quantity
                }
                records.append(record)
    else:
        print(f"Skipping file without 'auctions' key: {os.path.basename(filepath)}")
    return records

def winsorized_mean(data, limits):
    lower, upper = limits
    
    # Sort the data
    sorted_data = data.sort_values()
    
    # Calculate the cutoff points
    lower_cutoff = np.percentile(sorted_data, lower)
    upper_cutoff = np.percentile(sorted_data, upper)
    
    # Winsorize the data
    winsorized_data = np.clip(sorted_data, lower_cutoff, upper_cutoff)

    # Calculate the mean of the winsorized data
    return winsorized_data.mean()

def compute_average_prices(records):
    limits = (1, 75)  # Winsorize 10% from each end
    # if we have less than 5 records, return median

    if len(records) <= 5 and len(records) > 1:
        return np.median(records)
    
    # if we have 5 or more records, return the winsorized mean
    return winsorized_mean(records, limits)

def calculate_and_save_average_prices(records, output_path, mode='a'):
    if not records:
        print("No records to process.")
        return
    
    df = pd.DataFrame(records)
    
    if 'datetime' not in df.columns or 'item_id' not in df.columns or 'price' not in df.columns:
        print("One or more required columns are missing in the DataFrame")
        return
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.floor('h') 

    average_prices = df.groupby(['hour', 'item_id'])['price'].apply(lambda x: compute_average_prices(x)).reset_index()
    average_prices = average_prices.rename(columns={'hour': 'datetime'})
    average_prices['datetime'] = average_prices['datetime'].dt.strftime('%Y-%m-%d %H:00:00')
    
    header = True if mode == 'w' else False
    average_prices.to_csv(output_path, index=False, mode=mode, header=header)

def main():
    all_file_paths = []
    
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                all_file_paths.append(filepath)
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    for i, filepath in enumerate(tqdm(all_file_paths)):
        records = process_json_file(filepath)
        if records:
            calculate_and_save_average_prices(records, output_path, mode='a' if i > 0 else 'w')

if __name__ == "__main__":
    main()
