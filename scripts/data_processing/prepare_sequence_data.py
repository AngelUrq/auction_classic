import os, json, torch, time
import argparse
import pandas as pd
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

MAX_BONUSES = 9
MAX_MODIFIERS = 11

exclude_first_times = [
    '09-03-2025', # we should exclude items that appeared in the first two days
    '10-03-2025',
    '22-03-2025', # we should exclude items that appeared in the last two days
    '23-03-2025',
    '24-03-2025',
    '25-03-2025', 
    '26-03-2025',
    '20-08-2025',
    '21-08-2025',
    '22-08-2025', # we should exclude items that appeared in the last day and all after that
]

# Use the last date in the exclude_first_times array
last_exclude_date = datetime.strptime(exclude_first_times[-1], '%d-%m-%Y')

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
    
    total_files = len(file_info)
    print(f'Processing all {total_files} files')

    csv_file_path = os.path.join(args.output_dir, 'indices.csv')
    parquet_file_path = os.path.join(args.output_dir, 'indices.parquet')
    
    # Remove existing CSV file to start fresh
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    if not os.path.exists(os.path.join(args.output_dir, h5_filename)):
        print(f'Creating {h5_filename}')

        with h5py.File(os.path.join(args.output_dir, h5_filename), 'w') as f:
            pass

    for filepath in tqdm(list(file_info.keys())):
        prediction_time = file_info[filepath]
        
        # Skip files that come after the last exclude date
        if prediction_time.date() > last_exclude_date.date() or prediction_time.strftime('%d-%m-%Y') in exclude_first_times:
            print(f'Skipping {filepath}')
            continue
            
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
        
        for auction in auctions:
            auction_id = str(auction['id'])
            item_index = item_to_idx[str(auction['item']['id'])]
            bid = auction.get('bid', 0) / 10000.0
            buyout = auction.get('buyout', 0) / 10000.0
            time_left = auction['time_left']
            quantity = auction['quantity']
            context = context_to_idx[str(auction['item'].get('context', 0))]
            bonus_lists = [bonus_to_idx[str(bonus)] for bonus in auction['item'].get('bonus_lists', [])][:MAX_BONUSES]
            modifiers = auction['item'].get('modifiers', [])[:MAX_MODIFIERS]

            modifier_types = [modtype_to_idx[str(modifier['type'])] for modifier in modifiers]
            modifier_values = [modifier['value'] for modifier in modifiers]

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
            
            # Check if this time/hour group already exists
            if 'data' in h5_file[group_path]:
                print(f"Group {group_path} already exists. Skipping.")
                continue
                
            # Sort items by item_index for consistent ordering
            sorted_items = sorted(auctions_by_item.items())
            
            # Concatenate all auctions across all items
            all_auctions_list = []
            all_contexts_list = []
            all_bonus_lists = []
            all_modifier_types = []
            all_modifier_values = []
            
            # Track row splits using CSR/indptr format (1-D prefix sum)
            row_splits = [0]  # Start with 0
            sorted_item_ids = []
            
            for item_index, all_auctions in sorted_items:
                # Calculate statistics for this item
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
                
                # Add to concatenated lists
                auctions_data = np.array([auction[:6] for auction in all_auctions])
                contexts_data = np.array([auction[6] for auction in all_auctions])
                
                # Create fixed-size arrays directly
                bonus_sequences = [auction[7] for auction in all_auctions]
                modifier_type_sequences = [auction[8] for auction in all_auctions]
                modifier_value_sequences = [auction[9] for auction in all_auctions]
                
                bonus_lists_data = np.array([np.pad(seq, (0, max(0, MAX_BONUSES - len(seq))), 'constant')[:MAX_BONUSES] for seq in bonus_sequences])
                modifier_types_data = np.array([np.pad(seq, (0, max(0, MAX_MODIFIERS - len(seq))), 'constant')[:MAX_MODIFIERS] for seq in modifier_type_sequences])
                modifier_values_data = np.array([np.pad(seq, (0, max(0, MAX_MODIFIERS - len(seq))), 'constant')[:MAX_MODIFIERS] for seq in modifier_value_sequences])
                
                all_auctions_list.append(auctions_data)
                all_contexts_list.append(contexts_data)
                all_bonus_lists.append(bonus_lists_data)
                all_modifier_types.append(modifier_types_data)
                all_modifier_values.append(modifier_values_data)
                
                # Record row split using CSR format (cumulative sum)
                num_auctions = len(all_auctions)
                row_splits.append(row_splits[-1] + num_auctions)
                sorted_item_ids.append(item_index)
            
            # Concatenate all data
            if all_auctions_list:
                concatenated_data = np.concatenate(all_auctions_list, axis=0)
                concatenated_contexts = np.concatenate(all_contexts_list, axis=0)
                concatenated_bonus_lists = np.concatenate(all_bonus_lists, axis=0)
                concatenated_modifier_types = np.concatenate(all_modifier_types, axis=0)
                concatenated_modifier_values = np.concatenate(all_modifier_values, axis=0)
                
                # Determine chunking parameters
                total_rows = concatenated_data.shape[0]
                rows_chunk = min(4096, total_rows)  # Use 4096 or total rows if smaller
                
                # Store concatenated data with compression and chunking
                # Big row-major arrays with compression
                h5_file[group_path].create_dataset(
                    'data', 
                    data=concatenated_data,
                    compression='lzf',
                    chunks=(rows_chunk, concatenated_data.shape[1])
                )
                h5_file[group_path].create_dataset(
                    'contexts', 
                    data=concatenated_contexts,
                    compression='lzf',
                    chunks=(rows_chunk,)
                )
                h5_file[group_path].create_dataset(
                    'bonus_lists', 
                    data=concatenated_bonus_lists,
                    compression='lzf',
                    chunks=(rows_chunk, concatenated_bonus_lists.shape[1])
                )
                h5_file[group_path].create_dataset(
                    'modifier_types', 
                    data=concatenated_modifier_types,
                    compression='lzf',
                    chunks=(rows_chunk, concatenated_modifier_types.shape[1])
                )
                h5_file[group_path].create_dataset(
                    'modifier_values', 
                    data=concatenated_modifier_values,
                    compression='lzf',
                    chunks=(rows_chunk, concatenated_modifier_values.shape[1])
                )
                
                # Store row splits (CSR format) and sorted item IDs with light compression
                # Small index arrays
                h5_file[group_path].create_dataset(
                    'row_splits', 
                    data=np.array(row_splits, dtype=np.int32),
                    compression='lzf',
                    chunks=(len(row_splits),)
                )
                h5_file[group_path].create_dataset(
                    'item_ids', 
                    data=np.array(sorted_item_ids, dtype=np.int32),
                    compression='lzf',
                    chunks=(len(sorted_item_ids),)
                )

        if auction_indices:
            new_df = pd.DataFrame(auction_indices, columns=[
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
            
            # Append to CSV file (creates header on first write)
            new_df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
    
    # Convert CSV to parquet at the end
    if os.path.exists(csv_file_path):
        print('Converting CSV to parquet...')
        final_df = pd.read_csv(csv_file_path)
        final_df.to_parquet(parquet_file_path, index=False)
        os.remove(csv_file_path)  # Clean up CSV file
        print(f'Created {parquet_file_path} with {len(final_df)} records')
            
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
