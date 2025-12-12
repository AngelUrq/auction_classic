#!/usr/bin/env python3
import os, json, time
import argparse
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import gc
from collections import defaultdict, OrderedDict
from tqdm import tqdm

TIME_LEFT_TO_INT = {
    'VERY_LONG': 48.0,
    'LONG': 12.0,
    'MEDIUM': 2.0,
    'SHORT': 0.5,
}

MAX_BONUSES = 9
MAX_MODIFIERS = 11
MAX_SEQUENCE_LENGTH = 1024
ROW_CHUNK = 1024

# We exclude the first times of these days because their data is not complete.
# current_hours is not available for these days as you need to wait 48 hours to get it.
# We also exclude the last days because if you publish an auction in the last day, you can't get the hours_on_sale.
exclude_first_times = ['01-06-2025', '02-06-2025', '30-07-2025', '31-07-2025']
last_exclude_date = datetime.strptime(exclude_first_times[-1], '%d-%m-%Y')


def pad_or_truncate_bonuses(bonus_ids, bonus_to_idx):
    mapped = [int(bonus_to_idx[str(b)]) for b in bonus_ids][:MAX_BONUSES]
    if len(mapped) < MAX_BONUSES:
        mapped += [0] * (MAX_BONUSES - len(mapped))
    return np.asarray(mapped, dtype=np.int32)


def pad_or_truncate_modifiers(modifiers, modtype_to_idx):
    modifiers = modifiers[:MAX_MODIFIERS]
    types = [int(modtype_to_idx[str(m['type'])]) for m in modifiers]
    values = [float(m['value']) for m in modifiers]
    if len(types) < MAX_MODIFIERS:
        types += [0] * (MAX_MODIFIERS - len(types))
        values += [0.0] * (MAX_MODIFIERS - len(values))
    return (
        np.asarray(types, dtype=np.int32),
        np.asarray(values, dtype=np.float32)
    )


def _check_item_datasets(h5_file, item_id_str):
    grp_root = h5_file.require_group('items')
    grp = grp_root.require_group(item_id_str)

    if 'data' not in grp:
        grp.create_dataset('data', shape=(0, 6), maxshape=(None, 6), dtype='float32', chunks=(ROW_CHUNK, 6))
    if 'contexts' not in grp:
        grp.create_dataset('contexts', shape=(0,), maxshape=(None,), dtype='int32', chunks=(ROW_CHUNK,))
    if 'bonus_lists' not in grp:
        grp.create_dataset('bonus_lists', shape=(0, MAX_BONUSES), maxshape=(None, MAX_BONUSES), dtype='int32', chunks=(ROW_CHUNK, MAX_BONUSES))
    if 'modifier_types' not in grp:
        grp.create_dataset('modifier_types', shape=(0, MAX_MODIFIERS), maxshape=(None, MAX_MODIFIERS), dtype='int32', chunks=(ROW_CHUNK, MAX_MODIFIERS))
    if 'modifier_values' not in grp:
        grp.create_dataset('modifier_values', shape=(0, MAX_MODIFIERS), maxshape=(None, MAX_MODIFIERS), dtype='float32', chunks=(ROW_CHUNK, MAX_MODIFIERS))

    return grp


def _append_item_block(grp, data_h, contexts_h, bonus_lists_h, modifier_types_h, modifier_values_h):
    n = data_h.shape[0]
    old = grp['data'].shape[0]
    new = old + n

    grp['data'].resize((new, 6))
    grp['contexts'].resize((new,))
    grp['bonus_lists'].resize((new, MAX_BONUSES))
    grp['modifier_types'].resize((new, MAX_MODIFIERS))
    grp['modifier_values'].resize((new, MAX_MODIFIERS))

    grp['data'][old:new, :] = data_h
    grp['contexts'][old:new] = contexts_h
    grp['bonus_lists'][old:new, :] = bonus_lists_h
    grp['modifier_types'][old:new, :] = modifier_types_h
    grp['modifier_values'][old:new, :] = modifier_values_h
    
    return old, n


def process_auctions(args):
    print('Processing auctions...', flush=True)

    data_dir = args.data_dir
    mappings_dir = args.mappings_dir
    output_dir = args.output_dir

    h5_path = os.path.join(output_dir, 'sequences.h5')
    csv_path = os.path.join(output_dir, 'indices.csv')
    parquet_path = os.path.join(output_dir, 'indices.parquet')

    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Load mappings
    with open(args.timestamps, 'r') as f: timestamps = json.load(f)
    with open(os.path.join(mappings_dir, 'item_to_idx.json'), 'r') as f: item_to_idx = json.load(f)
    with open(os.path.join(mappings_dir, 'context_to_idx.json'), 'r') as f: context_to_idx = json.load(f)
    with open(os.path.join(mappings_dir, 'bonus_to_idx.json'), 'r') as f: bonus_to_idx = json.load(f)
    with open(os.path.join(mappings_dir, 'modtype_to_idx.json'), 'r') as f: modtype_to_idx = json.load(f)

    # Discover files and group by date
    files = []
    for root, _, fs in os.walk(data_dir):
        for fn in fs:
            fp = os.path.join(root, fn)
            dt = datetime.strptime(fn.split('.')[0], '%Y%m%dT%H')
            files.append((fp, dt))
    files.sort(key=lambda x: x[1])

    files_by_day = OrderedDict()
    for fp, dt in files:
        key = dt.strftime('%Y-%m-%d')
        files_by_day.setdefault(key, []).append((fp, dt))

    if not os.path.exists(h5_path):
        with h5py.File(h5_path, 'w'): pass

    header_needed = not os.path.exists(csv_path)

    # Process per day with tqdm
    for day_key, day_files in tqdm(list(files_by_day.items()), desc='days'):
        day_files = [(fp, dt) for (fp, dt) in day_files
                     if not (dt.date() > last_exclude_date.date() or dt.strftime('%d-%m-%Y') in exclude_first_times)]
        if not day_files:
            tqdm.write(f"[day {day_key}] skipped (policy filtered)")
            continue

        day_items = defaultdict(lambda: {
            'records': [], 'lengths': [], 'stats': [],
            'data_blocks': [], 'contexts_blocks': [],
            'bonus_blocks': [], 'modifier_types_blocks': [], 'modifier_values_blocks': [],
        })

        # Accumulate each hour in the day
        for filepath, prediction_time in tqdm(day_files, desc=f'{day_key} hours', leave=False):
            try:
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
            except json.JSONDecodeError:
                tqdm.write(f"[bad-json] Skipping {filepath}")
                continue

            auctions = json_data['auctions']
            record_str = prediction_time.strftime('%Y-%m-%d %H:00:00')
            print(record_str)

            auctions_by_item = {}
            for auction in auctions:
                if 'pet_species_id' in auction['item']:
                    continue

                auction_id = str(auction['id'])
                item_index = int(item_to_idx[str(auction['item']['id'])])
                bid = float(auction.get('bid', 0)) / 10000.0
                buyout = float(auction.get('buyout', 0)) / 10000.0
                time_left = float(TIME_LEFT_TO_INT[auction['time_left']])
                quantity = float(auction['quantity'])
                context = int(context_to_idx[str(auction['item'].get('context', 0))])

                bonus_arr = pad_or_truncate_bonuses(auction['item'].get('bonus_lists', []), bonus_to_idx)
                modifier_types, modifier_values = pad_or_truncate_modifiers(auction['item'].get('modifiers', []), modtype_to_idx)

                first_appearance = datetime.strptime(timestamps[auction_id]['first_appearance'], '%Y-%m-%d %H:%M:%S')
                last_appearance  = datetime.strptime(timestamps[auction_id]['last_appearance'],  '%Y-%m-%d %H:%M:%S')
                current_hours = (prediction_time - first_appearance).total_seconds() / 3600.0
                hours_on_sale = (last_appearance - prediction_time).total_seconds() / 3600.0

                row_data = np.array([bid, buyout, quantity, time_left, current_hours, hours_on_sale], dtype=np.float32)

                if item_index not in auctions_by_item:
                    auctions_by_item[item_index] = {
                        'data': [], 'contexts': [],
                        'bonus_lists': [], 'modifier_types': [], 'modifier_values': []
                    }

                auctions_by_item[item_index]['data'].append(row_data)
                auctions_by_item[item_index]['contexts'].append(context)
                auctions_by_item[item_index]['bonus_lists'].append(bonus_arr)
                auctions_by_item[item_index]['modifier_types'].append(modifier_types)
                auctions_by_item[item_index]['modifier_values'].append(modifier_values)

            for item_index, pack in auctions_by_item.items():
                data_h = np.vstack(pack['data']).astype(np.float32, copy=False)
                contexts_h = np.asarray(pack['contexts'], dtype=np.int32)
                bonus_lists_h = np.vstack(pack['bonus_lists']).astype(np.int32, copy=False)
                modifier_types_h = np.vstack(pack['modifier_types']).astype(np.int32, copy=False)
                modifier_values_h = np.vstack(pack['modifier_values']).astype(np.float32, copy=False)

                if data_h.shape[0] > MAX_SEQUENCE_LENGTH:
                    print(f'Capping {data_h.shape[0]} from item {item_index} to {MAX_SEQUENCE_LENGTH}')
                    data_h = data_h[:MAX_SEQUENCE_LENGTH]
                    contexts_h = contexts_h[:MAX_SEQUENCE_LENGTH]
                    bonus_lists_h = bonus_lists_h[:MAX_SEQUENCE_LENGTH]
                    modifier_types_h = modifier_types_h[:MAX_SEQUENCE_LENGTH]
                    modifier_values_h = modifier_values_h[:MAX_SEQUENCE_LENGTH]

                item_hours_on_sale = data_h[:, 5]
                item_current_hours = data_h[:, 4]
                stats_tuple = (
                    int(len(item_hours_on_sale)),
                    float(np.mean(item_hours_on_sale)), float(np.std(item_hours_on_sale)),
                    float(np.min(item_hours_on_sale)), float(np.max(item_hours_on_sale)),
                    float(np.mean(item_current_hours)), float(np.std(item_current_hours)),
                    float(np.min(item_current_hours)), float(np.max(item_current_hours)),
                )

                buf = day_items[item_index]
                buf['records'].append(record_str)
                buf['lengths'].append(int(data_h.shape[0]))
                buf['data_blocks'].append(data_h)
                buf['contexts_blocks'].append(contexts_h)
                buf['bonus_blocks'].append(bonus_lists_h)
                buf['modifier_types_blocks'].append(modifier_types_h)
                buf['modifier_values_blocks'].append(modifier_values_h)
                buf['stats'].append(stats_tuple)

            del auctions_by_item, auctions, json_data
            gc.collect()

        # Write once per item for the day + emit indices per hour
        indices_rows = []
        with h5py.File(h5_path, 'a') as h5f:
            for item_index, buf in day_items.items():
                grp = _check_item_datasets(h5f, str(item_index))

                data_day = np.concatenate(buf['data_blocks'], axis=0)
                contexts_day = np.concatenate(buf['contexts_blocks'], axis=0)
                bonus_lists_day = np.concatenate(buf['bonus_blocks'], axis=0)
                modifier_types_day = np.concatenate(buf['modifier_types_blocks'], axis=0)
                modifier_values_day = np.concatenate(buf['modifier_values_blocks'], axis=0)

                base_start, _ = _append_item_block(
                    grp, data_day, contexts_day, bonus_lists_day, modifier_types_day, modifier_values_day
                )

                prefix = 0
                for rec, stats in zip(buf['records'], buf['stats']):
                    g_len = stats[0]
                    indices_rows.append((
                        rec, int(item_index),
                        stats[0], stats[1], stats[2], stats[3], stats[4],
                        stats[5], stats[6], stats[7], stats[8],
                        np.uint64(base_start + prefix), np.uint32(g_len),
                    ))
                    prefix += g_len

        if indices_rows:
            cols = [
                'record', 'item_index',
                'g_hours_on_sale_len', 'g_hours_on_sale_mean', 'g_hours_on_sale_std',
                'g_hours_on_sale_min', 'g_hours_on_sale_max',
                'g_current_hours_mean', 'g_current_hours_std',
                'g_current_hours_min', 'g_current_hours_max',
                'start', 'length'
            ]
            pd.DataFrame(indices_rows, columns=cols).to_csv(
                csv_path, mode='a', header=header_needed, index=False
            )
            header_needed = False

        del day_items, indices_rows
        gc.collect()

    if os.path.exists(csv_path):
        print('Converting CSV to Parquet...', flush=True)
        gc.collect()
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False)
        print(f'Created {parquet_path} with {len(df)} records', flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Per-item HDF5 writer with 24h batching + indices CSV→Parquet')
    parser.add_argument('--data_dir', type=str, default='data/tww/auctions/')
    parser.add_argument('--timestamps', type=str, default='generated/timestamps.json')
    parser.add_argument('--output_dir', type=str, default='generated')
    parser.add_argument('--mappings_dir', type=str, default='generated/mappings/')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    process_auctions(args)
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds", flush=True)
