import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Analyze sequence length statistics from dataloader cache")
    parser.add_argument("--max-hours-back", type=int, default=72, help="The max_hours_back configuration to analyze")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Which dataset split to analyze")
    parser.add_argument("--cache-dir", type=str, default="generated/cache", help="Path to the dataloader cache directory")
    args = parser.parse_args()

    cache_file = os.path.join(args.cache_dir, f"bucket_lengths_{args.split}_{args.max_hours_back}.npy")

    if not os.path.exists(cache_file):
        print(f"Error: Cache file not found at {cache_file}")
        print(f"Please ensure you have run the Dataloader at least once with max_hours_back={args.max_hours_back}")
        return

    print(f"Loading cached lengths from {cache_file}...")
    lengths = np.load(cache_file)
    
    total_sequences = len(lengths)
    if total_sequences == 0:
        print("Error: The cache file is empty.")
        return

    print("=" * 50)
    print(f"Sequence Length Statistics (max_hours_back={args.max_hours_back}, split={args.split})")
    print("=" * 50)
    print(f"Total Sequences: {total_sequences:,}")
    print(f"Mean Length:     {np.mean(lengths):.2f}")
    print(f"Std Dev:         {np.std(lengths):.2f}")
    print(f"Min Length:      {np.min(lengths)}")
    print(f"Max Length:      {np.max(lengths)}")
    print("-" * 50)
    
    print("Quantiles:")
    percentiles = [1, 5, 25, 50, 75, 95, 99, 99.9]
    quant_vals = np.percentile(lengths, percentiles)
    for p, val in zip(percentiles, quant_vals):
        print(f"  {p:>4}%: {int(val):>5}")
        
    import matplotlib.pyplot as plt

    print("-" * 50)
    print("Generating Matplotlib Histogram...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Linear Y-axis
    ax1.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title(f'Linear Scale (max_hours_back={args.max_hours_back})')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Log Y-axis
    ax2.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_title(f'Log Scale (max_hours_back={args.max_hours_back})')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Frequency (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print("=" * 50)


if __name__ == "__main__":
    main()
