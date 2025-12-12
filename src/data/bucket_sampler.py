import logging
import numpy as np
import os
from datetime import datetime, timedelta
from torch.utils.data import Sampler
from typing import Iterator, List

logger = logging.getLogger(__name__)


class BucketBatchSampler(Sampler[List[int]]):
    """
    Sampler that groups sequences by length into buckets, then creates batches
    from within each bucket to minimize padding waste.
    
    Args:
        lengths: Array of sequence lengths for each sample
        batch_size: Number of samples per batch
        num_buckets: Number of length buckets
        shuffle: Whether to shuffle within buckets and shuffle bucket order
        drop_last: Whether to drop incomplete batches
    """
    
    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        num_buckets: int = 20,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Create bucket boundaries using percentiles for even distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        self.bucket_boundaries = np.percentile(self.lengths, percentiles)
        
        # Assign each sample to a bucket
        self.bucket_indices = self._assign_buckets()
        
    def _assign_buckets(self) -> List[List[int]]:
        """Assign each sample index to a bucket based on its length."""
        buckets = [[] for _ in range(self.num_buckets)]
        
        for idx, length in enumerate(self.lengths):
            bucket_id = np.searchsorted(self.bucket_boundaries[1:], length, side='right')
            bucket_id = min(bucket_id, self.num_buckets - 1)
            buckets[bucket_id].append(idx)
        
        return buckets
    
    def __iter__(self) -> Iterator[List[int]]:
        # Copy and optionally shuffle within each bucket
        buckets = []
        for bucket in self.bucket_indices:
            bucket = list(bucket)
            if self.shuffle:
                np.random.shuffle(bucket)
            buckets.append(bucket)
        
        # Create batches from each bucket
        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batch order
        if self.shuffle:
            np.random.shuffle(all_batches)
        
        yield from all_batches
    
    def __len__(self) -> int:
        total = 0
        for bucket in self.bucket_indices:
            if self.drop_last:
                total += len(bucket) // self.batch_size
            else:
                total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total


def compute_sequence_lengths(
    pairs_df,
    idx_map: dict,
    max_hours_back: int,
) -> np.ndarray:
    """
    Compute the exact sequence length for each sample by checking
    data availability across the lookback window.
    """
    from tqdm import tqdm
    
    lengths = np.zeros(len(pairs_df), dtype=np.int32)
    
    for i, row in enumerate(tqdm(pairs_df.itertuples(index=False), 
                                  total=len(pairs_df), 
                                  desc="Computing sequence lengths")):
        ts = datetime.strptime(row.record, "%Y-%m-%d %H:%M:%S")
        item_idx = int(row.item_index)
        
        total_length = 0
        for hours_back in range(max_hours_back + 1):
            t = ts - timedelta(hours=hours_back)
            k = t.strftime("%Y-%m-%d %H:00:00")
            key = (item_idx, k)
            if key in idx_map:
                _, length = idx_map[key]
                total_length += length
        
        lengths[i] = total_length
    
    return lengths


def get_lengths(
    pairs_df,
    idx_map: dict,
    max_hours_back: int,
    cache_dir: str = "generated",
    split: str = "train",
) -> np.ndarray:
    """
    Load cached lengths if available, otherwise compute and save them.
    
    Args:
        pairs_df: DataFrame with 'record', 'item_index' columns
        idx_map: Dict mapping (item_index, timestamp_key) -> (start, length)
        max_hours_back: Number of hours to look back
        cache_dir: Directory to store cached lengths
        split: Name of the data split (e.g., 'train', 'val') for separate caching
        
    Returns:
        Array of sequence lengths for each sample
    """
    cache_path = os.path.join(cache_dir, f"bucket_lengths_{split}_{max_hours_back}.npy")
    
    if os.path.exists(cache_path):
        lengths = np.load(cache_path)
        if len(lengths) == len(pairs_df):
            logger.info(f"Loaded cached lengths from {cache_path}")
            return lengths
        else:
            logger.info(f"Cache size mismatch ({len(lengths)} vs {len(pairs_df)}), recomputing...")
    
    logger.info(f"Computing sequence lengths for max_hours_back={max_hours_back}...")
    lengths = compute_sequence_lengths(pairs_df, idx_map, max_hours_back)
    
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, lengths)
    logger.info(f"Saved lengths to {cache_path}")
    
    return lengths

