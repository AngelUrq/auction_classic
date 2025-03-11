import os, json, torch, time
import argparse
import csv
import numpy as np
import h5py
from datetime import datetime
from tqdm import tqdm

time_left_to_int = {
    'VERY_LONG': 48,
    'LONG': 12,
    'MEDIUM': 2,
    'SHORT': 0.5
}

exclude_first_times = [
    '02-01-2024',
    '03-01-2024',
    '28-01-2024',
    '29-01-2024',
    '30-01-2024',
    '01-03-2024',
    '02-03-2024',
    '23-06-2024',
    '24-06-2024',
    '23-07-2024',
    '24-07-2024',
    '03-08-2024',
    '04-08-2024',
    '11-08-2024',
    '12-08-2024',
    '13-08-2024',
    '15-08-2024',
    '16-08-2024',
    '13-09-2024',
    '14-09-2024',
    '24-09-2024',
    '25-09-2024',
]

def process_auctions(args):
    print('Processing auctions...')
    file_info = {}
    data_dir = args.data_dir
    h5_filename = 'sequences.h5'

    with open(args.timestamps, 'r') as f:
        timestamps = json.load(f)

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            date = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')
            file_info[filepath] = date

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}

    if not os.path.exists(os.path.join(args.output_dir, 'auction_indices.csv')):
        print('Creating auction_indices.csv')

        with open(os.path.join(args.output_dir, 'auction_indices.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'record', 
                'item_id', 
                'group_hours_on_sale_len',
                'group_hours_on_sale_mean',
                'group_hours_on_sale_std',
                'group_hours_on_sale_min',
                'group_hours_on_sale_max',
                'group_hours_since_first_appearance_mean',
                'group_hours_since_first_appearance_std',
                'group_hours_since_first_appearance_min',
                'group_hours_since_first_appearance_max',
                'expansion'
            ])

    if not os.path.exists(os.path.join(args.output_dir, h5_filename)):
        print(f'Creating {h5_filename}')

        with h5py.File(os.path.join(args.output_dir, h5_filename), 'w') as f:
            pass

    for filepath in tqdm(list(file_info.keys())):
        with open(filepath, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading file {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error loading file {filepath}: {e}")
                continue
                
        auctions = json_data['auctions']
        prediction_time = file_info[filepath]
        auctions_by_item = {}
        auction_indices = []

        if prediction_time.strftime('%d-%m-%Y') in exclude_first_times:
            continue

        for auction in auctions:
            auction_id = str(auction['id'])
            item_id = auction['item']['id']
            bid = auction['bid'] / 10000.0
            buyout = auction['buyout'] / 10000.0
            time_left = auction['time_left']
            quantity = auction['quantity']

            first_appearance = datetime.strptime(timestamps[auction_id]['first_appearance'], '%Y-%m-%d %H:%M:%S')
            last_appearance = datetime.strptime(timestamps[auction_id]['last_appearance'], '%Y-%m-%d %H:%M:%S')

            hours_since_first_appearance = (prediction_time - first_appearance).total_seconds() / 3600
            hours_on_sale = (last_appearance - prediction_time).total_seconds() / 3600

            processed_auction = [
                item_id,
                bid,
                buyout,
                quantity,
                time_left_to_int.get(time_left, 0),
                hours_since_first_appearance,
                hours_on_sale
            ]
            
            if item_id not in auctions_by_item:
                auctions_by_item[item_id] = []

            auctions_by_item[item_id].append(processed_auction)

        with h5py.File(os.path.join(args.output_dir, h5_filename), 'a') as h5_file:
            group_path = f"{prediction_time.strftime('%Y-%m-%d')}/{prediction_time.strftime('%H')}"
            if group_path not in h5_file:
                h5_file.create_group(group_path)
                
            for item_id, auctions in auctions_by_item.items():
                item_hours_on_sale = np.array([auction[6] for auction in auctions])
                item_hours_since_first_appearance = np.array([auction[5] for auction in auctions])

                group_hours_on_sale_mean = round(np.mean(item_hours_on_sale), 2)
                group_hours_on_sale_std = round(np.std(item_hours_on_sale), 2)
                group_hours_on_sale_min = round(np.min(item_hours_on_sale), 2)
                group_hours_on_sale_max = round(np.max(item_hours_on_sale), 2)
                group_hours_on_sale_len = len(item_hours_on_sale)

                group_hours_since_first_appearance_mean = round(np.mean(item_hours_since_first_appearance), 2)
                group_hours_since_first_appearance_std = round(np.std(item_hours_since_first_appearance), 2)
                group_hours_since_first_appearance_min = round(np.min(item_hours_since_first_appearance), 2)
                group_hours_since_first_appearance_max = round(np.max(item_hours_since_first_appearance), 2)

                expansion = 'cata' if prediction_time > datetime(2024, 6, 1) else 'wotlk'

                auction_indices.append((
                    prediction_time.strftime('%Y-%m-%d %H:%M:%S'), 
                    item_id, 
                    group_hours_on_sale_len, 
                    group_hours_on_sale_mean, 
                    group_hours_on_sale_std, 
                    group_hours_on_sale_min, 
                    group_hours_on_sale_max,
                    group_hours_since_first_appearance_mean,
                    group_hours_since_first_appearance_std,
                    group_hours_since_first_appearance_min,
                    group_hours_since_first_appearance_max,
                    expansion
                ))
                                
                dataset_name = f'{item_id}'
                if dataset_name in h5_file[group_path]:
                    print(f"Dataset {dataset_name} already exists in group {group_path}. Skipping.")
                    continue

                h5_file[group_path].create_dataset(dataset_name, data=np.array(auctions))

        with open(os.path.join(args.output_dir, 'auction_indices.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(auction_indices)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check JSON files in auctions folder')
    parser.add_argument('--data_dir', type=str, help='Path to the auctions folder', default=os.path.abspath('data/auctions/'))
    parser.add_argument('--timestamps', type=str, help='Path to the timestamps JSON file', default=os.path.abspath('generated/timestamps.json'))
    parser.add_argument('--output_dir', type=str, help='Path to the output folder', default=os.path.abspath('generated/'))
    args = parser.parse_args()

    start_time = time.time()
    process_auctions(args)
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
