#!/usr/bin/env python3
"""
Compute statistics for listing_duration derived from timestamps.json.

listing_duration = last_appearance - first_appearance  (in hours)

Usage:
    python scripts/analyze/listing_duration_stats.py
    python scripts/analyze/listing_duration_stats.py --timestamps generated/timestamps.json
    python scripts/analyze/listing_duration_stats.py --output figures/listing_duration.png
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_durations(timestamps_path: str) -> np.ndarray:
    """
    Parse timestamps.json and return listing_duration = last - first (hours)
    for every auction.
    """
    print(f"Loading {timestamps_path} ...", flush=True)
    with open(timestamps_path, "r") as f:
        data = json.load(f)

    fmt = "%Y-%m-%d %H:%M:%S"
    durations = []
    for entry in tqdm(data.values(), desc="Computing durations"):
        first = datetime.strptime(entry["first_appearance"], fmt)
        last  = datetime.strptime(entry["last_appearance"],  fmt)
        durations.append((last - first).total_seconds() / 3600.0)

    return np.array(durations, dtype=np.float64)


def print_stats(d: np.ndarray) -> None:
    """Print a comprehensive statistical summary to stdout."""
    quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q_values = np.percentile(d, quantiles)

    buckets = [
        ("0 h",        d == 0),
        ("(0, 2] h",   (d > 0)  & (d <= 2)),
        ("(2, 12] h",  (d > 2)  & (d <= 12)),
        ("(12, 24] h", (d > 12) & (d <= 24)),
        ("(24, 48] h", (d > 24) & (d <= 48)),
        ("> 48 h",     d > 48),
    ]

    print("\n" + "=" * 60)
    print("  listing_duration  STATISTICS  (hours)")
    print("=" * 60)
    print(f"  Observations      : {len(d):>12,}")
    print(f"  Mean              : {d.mean():>12.2f} h")
    print(f"  Std               : {d.std():>12.2f} h")
    print(f"  Min               : {d.min():>12.2f} h")
    print(f"  Max               : {d.max():>12.2f} h")
    print()
    print("  Quantiles:")
    for q, v in zip(quantiles, q_values):
        print(f"    p{q:>3}            : {v:>12.2f} h")
    print()
    print("  Duration buckets:")
    for label, mask in buckets:
        n = mask.sum()
        print(f"    {label:<16}: {n:>12,}  ({100*n/len(d):5.1f} %)")
    print()
    max_h = int(d.max())
    print(f"  Per-hour counts (0–{max_h} h):")
    for h in range(max_h + 1):
        n = (d == h).sum()
        bar = "#" * int(40 * n / len(d))
        print(f"    {h:>3} h: {n:>10,}  ({100*n/len(d):5.1f} %)  {bar}")
    print("=" * 60)


def plot_histogram(d: np.ndarray, output_path: str | None) -> None:
    """Plot duration histogram on linear and log-y scales side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("listing_duration distribution", fontsize=14)

    clip = 200
    common = dict(bins=100, edgecolor="none", alpha=0.8, color="steelblue")

    # Left: linear y-axis, clipped tail
    axes[0].hist(np.clip(d, 0, clip), **common)
    axes[0].axvline(np.median(d), color="gold", linewidth=1.5,
                    linestyle="-", label=f"median ({np.median(d):.1f} h)")
    axes[0].set_xlabel("Duration (h)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Linear  [clipped at {clip} h]")
    axes[0].legend(fontsize=8)

    # Right: log y-axis, full range
    axes[1].hist(d, **common)
    axes[1].set_yscale("log")
    axes[1].axvline(np.median(d), color="gold", linewidth=1.5,
                    linestyle="-", label=f"median ({np.median(d):.1f} h)")
    axes[1].set_xlabel("Duration (h)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Log-Y  [full range]")
    axes[1].legend(fontsize=8)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"\nFigure saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="listing_duration statistics from timestamps.json")
    parser.add_argument("--timestamps", default="generated/timestamps.json")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path instead of showing interactively")
    args = parser.parse_args()

    durations = load_durations(args.timestamps)
    print_stats(durations)
    plot_histogram(durations, args.output)


if __name__ == "__main__":
    main()
