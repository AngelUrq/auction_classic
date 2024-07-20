import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

data_dir = 'auctions/'  
output_path = 'historical/historical_prices.csv'  

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
                record = {
                    'datetime': date,
                    'item_id': auction['item']['id'],
                    'price': buyout_in_gold
                }
                records.append(record)
    else:
        print(f"Skipping file without 'auctions' key: {os.path.basename(filepath)}")
    return records

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
    
    average_prices = df.groupby(['hour', 'item_id'])['price'].mean().reset_index()
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
        print(f"Processed file {i + 1} / {len(all_file_paths)}")

if __name__ == "__main__":
    main()
