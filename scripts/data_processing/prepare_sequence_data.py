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
    '09-03-2025', # we should exclude items that appeared in the first two days
    '10-03-2025',
    '22-03-2025', # we should exclude items that appeared in the last two days
    '23-03-2025',
    '25-03-2025', 
    '26-03-2025',
    '03-04-2025',
    '04-04-2025',
]

def pad_sequence(sequences, padding_value=0):
    sequences_np = [np.array(seq) for seq in sequences]
    
    # Find the max length among all sequences
    max_length = max(len(seq) for seq in sequences_np)
    
    # Get the shape of each sequence (excluding the length dimension)
    sample_shape = sequences_np[0].shape[1:] if len(sequences_np[0].shape) > 1 else ()

    # Create the output padded array
    padded_sequences = np.full((len(sequences_np), max_length) + sample_shape, padding_value, dtype=np.float32)
    
    # Fill in the actual sequences
    for i, seq in enumerate(sequences_np):
        length = len(seq)
        padded_sequences[i, :length] = seq
            
    return padded_sequences

def process_auctions(args):
    print('Processing auctions...')
    file_info = {}
    data_dir = args.data_dir
    h5_filename = 'sequences.h5'
    mappings_dir = args.mappings_dir

    with open(args.timestamps, 'r') as f:
        timestamps = json.load(f)

    with open(os.path.join(mappings_dir, 'item_to_idx.json'), 'r') as f:
        item_to_idx = json.load(f)

    with open(os.path.join(mappings_dir, 'context_to_idx.json'), 'r') as f:
        context_to_idx = json.load(f)
        
    with open(os.path.join(mappings_dir, 'bonus_to_idx.json'), 'r') as f:
        bonus_to_idx = json.load(f)

    with open(os.path.join(mappings_dir, 'modtype_to_idx.json'), 'r') as f:
        modtype_to_idx = json.load(f)

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
                'item_index', 
                'g_hours_on_sale_len',
                'g_hours_on_sale_mean',
                'g_hours_on_sale_std',
                'g_hours_on_sale_min',
                'g_hours_on_sale_max',
                'g_current_hours_mean',
                'g_current_hours_std',
                'g_current_hours_min',
                'g_current_hours_max'
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
            item_index = item_to_idx[str(auction['item']['id'])]
            bid = auction.get('bid', 0) / 10000.0
            buyout = auction.get('buyout', 0) / 10000.0
            time_left = auction['time_left']
            quantity = auction['quantity']
            context = context_to_idx[str(auction['item'].get('context', 0))]
            bonus_lists = [bonus_to_idx[str(bonus)] for bonus in auction['item'].get('bonus_lists', [])]
            modifiers = auction['item'].get('modifiers', [])

            modifier_types = []
            modifier_values = []

            for modifier in modifiers:
                modifier_types.append(modtype_to_idx[str(modifier['type'])])
                modifier_values.append(modifier['value'])

            if 'pet_species_id' in auction['item']:
                continue

            first_appearance = datetime.strptime(timestamps[auction_id]['first_appearance'], '%Y-%m-%d %H:%M:%S')
            last_appearance = datetime.strptime(timestamps[auction_id]['last_appearance'], '%Y-%m-%d %H:%M:%S')

            current_hours = (prediction_time - first_appearance).total_seconds() / 3600
            hours_on_sale = (last_appearance - prediction_time).total_seconds() / 3600

            processed_auction = [
                bid,
                buyout,
                quantity,
                time_left_to_int[time_left],
                current_hours,
                hours_on_sale,
                context,
                np.array(bonus_lists),
                np.array(modifier_types),
                np.array(modifier_values),
            ]
            
            if item_index not in auctions_by_item:
                auctions_by_item[item_index] = []

            auctions_by_item[item_index].append(processed_auction)

        with h5py.File(os.path.join(args.output_dir, h5_filename), 'a') as h5_file:
            group_path = f"{prediction_time.strftime('%Y-%m-%d')}/{prediction_time.strftime('%H')}"
            if group_path not in h5_file:
                h5_file.create_group(group_path)
                
            for item_index, all_auctions in auctions_by_item.items():
                item_hours_on_sale = np.array([auction[5] for auction in all_auctions])
                item_current_hours = np.array([auction[4] for auction in all_auctions])

                group_hours_on_sale_mean = round(np.mean(item_hours_on_sale), 2)
                group_hours_on_sale_std = round(np.std(item_hours_on_sale), 2)
                group_hours_on_sale_min = round(np.min(item_hours_on_sale), 2)
                group_hours_on_sale_max = round(np.max(item_hours_on_sale), 2)
                group_hours_on_sale_len = len(item_hours_on_sale)

                group_current_hours_mean = round(np.mean(item_current_hours), 2)
                group_current_hours_std = round(np.std(item_current_hours), 2)
                group_current_hours_min = round(np.min(item_current_hours), 2)
                group_current_hours_max = round(np.max(item_current_hours), 2)

                auction_indices.append((
                    prediction_time.strftime('%Y-%m-%d %H:%M:%S'), 
                    item_index, 
                    group_hours_on_sale_len, 
                    group_hours_on_sale_mean, 
                    group_hours_on_sale_std, 
                    group_hours_on_sale_min, 
                    group_hours_on_sale_max,
                    group_current_hours_mean,
                    group_current_hours_std,
                    group_current_hours_min,
                    group_current_hours_max
                ))
                                
                dataset_name = f'{item_index}'
                if dataset_name in h5_file[group_path]:
                    print(f"Dataset {dataset_name} already exists in file {group_path}. Skipping.")
                    continue

                auctions = np.array([auction[:6] for auction in all_auctions])
                contexts = np.array([auction[6] for auction in all_auctions])
                bonus_lists = pad_sequence([auction[7] for auction in all_auctions])
                modifier_types = pad_sequence([auction[8] for auction in all_auctions])
                modifier_values = pad_sequence([auction[9] for auction in all_auctions])
                
                buyout_prices = auctions[:, 1] 
                
                buyout_for_ranking = np.copy(buyout_prices)
                buyout_for_ranking[buyout_for_ranking == 0] = np.inf
            
                buyout_ranking = np.argsort(np.argsort(buyout_for_ranking)) + 1
                buyout_ranking[buyout_prices == 0] = 0
                
                h5_file[group_path].create_dataset(f'{dataset_name}/auctions', data=auctions)
                h5_file[group_path].create_dataset(f'{dataset_name}/contexts', data=contexts)
                h5_file[group_path].create_dataset(f'{dataset_name}/bonus_lists', data=bonus_lists)
                h5_file[group_path].create_dataset(f'{dataset_name}/modifier_types', data=modifier_types)
                h5_file[group_path].create_dataset(f'{dataset_name}/modifier_values', data=modifier_values)
                h5_file[group_path].create_dataset(f'{dataset_name}/buyout_ranking', data=buyout_ranking)

        with open(os.path.join(args.output_dir, 'auction_indices.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(auction_indices)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check JSON files in auctions folder')
    parser.add_argument('--data_dir', type=str, help='Path to the auctions folder', required=True)
    parser.add_argument('--timestamps', type=str, help='Path to the timestamps JSON file', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to the output folder', required=True)
    parser.add_argument('--mappings_dir', type=str, help='Path to the mappings folder', required=True)
    args = parser.parse_args()

    start_time = time.time()
    process_auctions(args)
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
