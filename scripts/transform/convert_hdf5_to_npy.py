#!/usr/bin/env python3
import os
import argparse
import pickle
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert sequences.h5 + indices to global .npy memmaps")
    parser.add_argument("--h5_path", type=str, default="generated/sequences.h5")
    parser.add_argument("--indices_path", type=str, default="generated/indices.parquet")
    parser.add_argument("--output_dir", type=str, default="generated/memmap")
    args = parser.parse_args()

    h5_path = args.h5_path
    indices_path = args.indices_path
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    idx_map_path = os.path.join(output_dir, "idx_map_global.pkl")
    if os.path.exists(idx_map_path):
        print(f"Skipping: {idx_map_path} already exists.")
        return

    print(f"Loading indices from {indices_path} ...", flush=True)
    df = pd.read_parquet(indices_path)
    df = df[["record", "item_index", "start", "length"]]

    item_max_end = defaultdict(int)

    for row in tqdm(df.itertuples(index=False), desc="Computing per-item max end", total=len(df)):
        item_idx = int(row.item_index)
        local_start = int(row.start)
        length = int(row.length)
        end = local_start + length
        if end > item_max_end[item_idx]:
            item_max_end[item_idx] = end

    item_list = sorted(item_max_end.keys())

    offset_by_item = {}
    running = 0
    for item_idx in item_list:
        offset_by_item[item_idx] = running
        running += item_max_end[item_idx]

    TOTAL_ROWS = running
    print(f"TOTAL_ROWS = {TOTAL_ROWS}", flush=True)

    data_mm = np.memmap(
        os.path.join(output_dir, "data.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(TOTAL_ROWS, 6),
    )
    contexts_mm = np.memmap(
        os.path.join(output_dir, "contexts.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(TOTAL_ROWS,),
    )
    bonus_ids_mm = np.memmap(
        os.path.join(output_dir, "bonus_ids.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(TOTAL_ROWS, 9),
    )
    modifier_types_mm = np.memmap(
        os.path.join(output_dir, "modifier_types.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(TOTAL_ROWS, 11),
    )
    modifier_values_mm = np.memmap(
        os.path.join(output_dir, "modifier_values.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(TOTAL_ROWS, 11),
    )

    print(f"\nReading from HDF5: {h5_path}\n", flush=True)

    with h5py.File(h5_path, "r") as f:
        for item_idx in tqdm(item_list, desc="Copying item groups"):
            base = offset_by_item[item_idx]
            N_item = item_max_end[item_idx]

            local_slice = slice(0, N_item)
            global_slice = slice(base, base + N_item)

            grp = f[f"items/{item_idx}"]

            data_mm[global_slice]            = grp["data"][local_slice]
            contexts_mm[global_slice]        = grp["contexts"][local_slice]
            bonus_ids_mm[global_slice]       = grp["bonus_ids"][local_slice]
            modifier_types_mm[global_slice]  = grp["modifier_types"][local_slice]
            modifier_values_mm[global_slice] = grp["modifier_values"][local_slice]

    data_mm.flush()
    contexts_mm.flush()
    bonus_ids_mm.flush()
    modifier_types_mm.flush()
    modifier_values_mm.flush()

    print("\nBuilding global index map...", flush=True)

    global_idx_map = {}

    for row in tqdm(df.itertuples(index=False), desc="Building global index map", total=len(df)):
        item_idx = int(row.item_index)
        record = row.record
        local_start = int(row.start)
        length = int(row.length)

        base = offset_by_item[item_idx]
        global_start = base + local_start

        global_idx_map[(item_idx, record)] = (global_start, length)

    out_path = os.path.join(output_dir, "idx_map_global.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(global_idx_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved global idx_map to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
