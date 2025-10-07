import torch
import h5py
import numpy as np
from datetime import datetime, timedelta

class AuctionDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_stats=None, path="../generated/sequences.h5", max_hours_back=0):
        self.pairs = pairs
        self.feature_stats = feature_stats
        self.max_hours_back = max_hours_back
        self.column_map = {
            "bid": 0,
            "buyout": 1,
            "quantity": 2,
            "time_left": 3,
            "current_hours": 4,
            "hours_on_sale": 5,
        }
        self.h5_file = h5py.File(path, "r")

    def __len__(self):
        return len(self.pairs)

    def __del__(self):
        self.h5_file.close()

    def _get_item_data_from_hour(self, target_group, item_idx):
        """
        Extract data for a specific item from an hour group using CSR indexing.
        Returns None if the item doesn't exist in that hour.
        """
        try:
            # Get the CSR indexing arrays
            row_splits = target_group["row_splits"][:]
            item_ids = target_group["item_ids"][:]
            
            # Find the item in the item_ids array using searchsorted
            item_position = np.searchsorted(item_ids, item_idx)
            if item_position >= len(item_ids) or item_ids[item_position] != item_idx:
                return None  # Item not found in this hour
            start_row = row_splits[item_position]
            end_row = row_splits[item_position + 1]
            
            # Extract the item's data from the concatenated arrays
            return {
                'auctions': torch.tensor(target_group["data"][start_row:end_row], dtype=torch.float32),
                'contexts': torch.tensor(target_group["contexts"][start_row:end_row], dtype=torch.int32).long(),
                'bonus_lists': torch.tensor(target_group["bonus_lists"][start_row:end_row], dtype=torch.int32).long(),
                'modifier_types': torch.tensor(target_group["modifier_types"][start_row:end_row], dtype=torch.int32).long(),
                'modifier_values': torch.tensor(target_group["modifier_values"][start_row:end_row], dtype=torch.float32),
            }
        except (KeyError, IndexError):
            return None

    def _get_temporal_data(self, base_timestamp, item_idx, hours_back):
        """
        Get auction data from a specific number of hours back from base_timestamp.
        When hours_back=0, returns current timestamp data.
        Returns None if data doesn't exist for that timestamp.
        """
        target_ts = base_timestamp - timedelta(hours=hours_back)
        target_path = f"{target_ts:%Y-%m-%d}/{target_ts:%H}"
        
        try:
            target_group = self.h5_file[target_path]
            item_data = self._get_item_data_from_hour(target_group, item_idx)
            
            if item_data is None:
                return None
            
            # Add hour_of_week to the data
            item_data['hour_of_week'] = target_ts.weekday() * 24 + target_ts.hour
            return item_data
            
        except KeyError:
            # Data doesn't exist for this timestamp
            return None

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        ts = datetime.strptime(pair["record"], "%Y-%m-%d %H:%M:%S")
        item_idx = pair["item_index"]
        
        # Collect all temporal data (including current time as hours_back=0)
        all_time_data = []
        actual_hours_back = []  # Track which hours actually have data
        
        # Get data from max_hours_back ago to current (0 hours back)
        for hours_back in range(self.max_hours_back, -1, -1):
            temporal_data = self._get_temporal_data(ts, item_idx, hours_back)
            
            if temporal_data is None:
                # For missing data, we need a reference to create padding
                # Use the current timestamp data (hours_back=0) as reference if available
                if hours_back == 0:
                    # If current data is missing, that's a real error
                    raise KeyError(f"Current timestamp data missing for {ts:%Y-%m-%d %H}:{item_idx}")
                
                # For historical data, skip hours with no auctions
                continue
            
            # Check if this hour actually has auctions (non-empty data)
            if temporal_data['auctions'].size(0) == 0:
                # Skip hours with no auctions
                continue
            
            all_time_data.append(temporal_data)
            actual_hours_back.append(hours_back)
        
        # Concatenate all temporal data (no padding needed - arrays are fixed-size)
        auctions = torch.cat([data['auctions'] for data in all_time_data], dim=0)
        contexts = torch.cat([data['contexts'] for data in all_time_data], dim=0)
        bonus_lists = torch.cat([data['bonus_lists'] for data in all_time_data], dim=0)
        modifier_types = torch.cat([data['modifier_types'] for data in all_time_data], dim=0)
        modifier_values = torch.cat([data['modifier_values'] for data in all_time_data], dim=0)
        
        # Create hour_of_week tensor for all timestamps
        hour_of_week_values = [data['hour_of_week'] for data in all_time_data]
        # Calculate repeat counts for each timestamp based on actual auction counts
        repeat_counts = [data['auctions'].size(0) for data in all_time_data]
        hour_of_week_tensor = torch.tensor(hour_of_week_values, dtype=torch.int32).repeat_interleave(torch.tensor(repeat_counts))
        
        # Create time_offset tensor using actual hours that have data
        time_offset_tensor = torch.tensor(actual_hours_back, dtype=torch.int32).repeat_interleave(torch.tensor(repeat_counts))

        # Extract hours_on_sale from the auctions data (it's the last column)
        hours_on_sale = auctions[:, self.column_map["hours_on_sale"]]
        auctions = auctions[:, :-1]

        auctions[:, self.column_map["bid"]] = torch.log1p(auctions[:, self.column_map["bid"]])
        auctions[:, self.column_map["buyout"]] = torch.log1p(auctions[:, self.column_map["buyout"]])
        modifier_values = torch.log1p(modifier_values)

        current_hours_raw = auctions[:, self.column_map["current_hours"]].clone()
        time_left_raw = auctions[:, self.column_map["time_left"]].clone()

        if self.feature_stats is not None:
            auctions = (auctions - self.feature_stats["means"][:5]) / (self.feature_stats["stds"][:5] + 1e-6)
            modifier_values = (modifier_values - self.feature_stats["modifiers_mean"]) / (self.feature_stats["modifiers_std"] + 1e-6)

        # Create item_index tensor for all temporal sequences
        item_index_tensor = torch.tensor(item_idx, dtype=torch.int32).repeat(auctions.size(0))
        
        # Use the pre-calculated hour_of_week_tensor
        y = hours_on_sale

        return {
            'auctions': auctions,
            'item_index': item_index_tensor,
            'contexts': contexts,
            'bonus_lists': bonus_lists,
            'modifier_types': modifier_types,
            'modifier_values': modifier_values,
            'hour_of_week': hour_of_week_tensor,
            'time_offset': time_offset_tensor,
            'current_hours_raw': current_hours_raw,
            'time_left_raw': time_left_raw,
            'target': y
        }
