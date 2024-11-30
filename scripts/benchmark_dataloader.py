import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def benchmark_dataset(dataset, batch_size=32, num_workers=0, num_epochs=1):
    """
    Benchmark dataset loading performance.
    
    Args:
        dataset: The dataset to benchmark
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for data loading
        num_epochs: Number of epochs to run the benchmark
        
    Returns:
        dict: Dictionary containing various timing statistics
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_auctions
    )
    
    # Initialize timing lists
    batch_times = []
    epoch_times = []
    
    print(f"\nBenchmarking with batch_size={batch_size}, num_workers={num_workers}")
    print(f"Dataset size: {len(dataset)} samples")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Use tqdm for progress tracking
        with tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (X, y, lengths) in enumerate(pbar):
                batch_start = time.time()
                
                # Simulate some basic processing to ensure data is actually loaded
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Update progress bar with current batch time
                pbar.set_postfix({
                    'batch_time': f'{batch_time:.4f}s',
                    'avg_time': f'{np.mean(batch_times):.4f}s'
                })
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
    # Calculate statistics
    stats = {
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'min_batch_time': np.min(batch_times),
        'max_batch_time': np.max(batch_times),
        'avg_epoch_time': np.mean(epoch_times),
        'std_epoch_time': np.std(epoch_times),
        'total_time': sum(epoch_times),
        'samples_per_second': len(dataset) * num_epochs / sum(epoch_times)
    }
    
    return stats

def run_benchmarks(dataset, batch_sizes=[32, 64, 128], num_workers_list=[0, 2, 4]):
    """
    Run benchmarks with different batch sizes and number of workers.
    
    Args:
        dataset: The dataset to benchmark
        batch_sizes: List of batch sizes to test
        num_workers_list: List of number of workers to test
    """
    results = {}
    
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            key = f"batch_{batch_size}_workers_{num_workers}"
            stats = benchmark_dataset(dataset, batch_size, num_workers)
            results[key] = stats
            
            print(f"\nResults for batch_size={batch_size}, num_workers={num_workers}:")
            print(f"Average batch loading time: {stats['avg_batch_time']:.4f}s ± {stats['std_batch_time']:.4f}s")
            print(f"Average epoch time: {stats['avg_epoch_time']:.2f}s")
            print(f"Samples per second: {stats['samples_per_second']:.2f}")
            print("-" * 80)
    
    return results

results = run_benchmarks(
    val_dataset,
    batch_sizes=[32, 128],
    num_workers_list=[0, 4]
)
