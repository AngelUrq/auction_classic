#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(repo_root))

from src.data.auction_dataset import AuctionDataset
from src.data.utils import collate_auctions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_pairs(
    indices_path: Path,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    """Load indices and optionally filter/slice before computing stats."""
    filters = []
    if date_start:
        filters.append(("record", ">=", date_start))
    if date_end:
        filters.append(("record", "<=", date_end))

    logger.info("Loading indices parquet for feature stats computation...")
    pairs = pd.read_parquet(
        indices_path,
        engine="pyarrow",
        filters=filters if filters else None,
    )

    logger.info("  Loaded %d rows after filtering/slicing", len(pairs))
    return pairs[["record", "item_index"]]


def _accumulate_batch(
    batch: dict,
    sum_features: torch.Tensor | None,
    sum_sq_features: torch.Tensor | None,
    modifier_sum: torch.Tensor | None,
    modifier_sum_sq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, int]:
    """Accumulate sums and squared sums for auctions and modifiers."""
    auctions = batch["auctions"]
    modifier_values = batch["modifier_values"]

    valid_mask = batch["item_index"] != 0

    auction_mask = valid_mask.unsqueeze(-1).expand_as(auctions)
    valid_auctions = auctions[auction_mask].reshape(-1, auctions.size(-1))

    if sum_features is None:
        sum_features = valid_auctions.sum(dim=0)
        sum_sq_features = (valid_auctions**2).sum(dim=0)
    else:
        sum_features += valid_auctions.sum(dim=0)
        sum_sq_features += (valid_auctions**2).sum(dim=0)

    total_count = valid_auctions.size(0)

    modifier_mask = valid_mask.unsqueeze(-1).expand_as(modifier_values)
    valid_modifiers = modifier_values[modifier_mask]

    if modifier_sum is None:
        modifier_sum = valid_modifiers.sum()
        modifier_sum_sq = (valid_modifiers**2).sum()
    else:
        modifier_sum += valid_modifiers.sum()
        modifier_sum_sq += (valid_modifiers**2).sum()

    modifier_count = valid_modifiers.size(0)

    return (
        sum_features,
        sum_sq_features,
        total_count,
        modifier_sum,
        modifier_sum_sq,
        modifier_count,
    )


def compute_feature_stats(
    dataloader: torch.utils.data.DataLoader, max_batches: int | None = None
) -> dict:
    """Compute per-feature mean/std for auctions and modifier values."""
    sum_features = None
    sum_sq_features = None
    total_count = 0

    modifier_sum = None
    modifier_sum_sq = None
    modifier_count = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Batches")):
        if max_batches is not None and i >= max_batches:
            break

        (
            sum_features,
            sum_sq_features,
            batch_count,
            modifier_sum,
            modifier_sum_sq,
            batch_modifier_count,
        ) = _accumulate_batch(
            batch, sum_features, sum_sq_features, modifier_sum, modifier_sum_sq
        )

        total_count += batch_count
        modifier_count += batch_modifier_count

    if total_count == 0 or modifier_count == 0:
        raise RuntimeError("No valid samples found while computing feature stats.")

    means = sum_features / total_count
    variances = sum_sq_features / total_count - means**2
    stds = torch.sqrt(torch.clamp(variances, min=1e-12))

    modifier_mean = modifier_sum / modifier_count
    modifier_variance = modifier_sum_sq / modifier_count - modifier_mean**2
    modifier_std = torch.sqrt(torch.clamp(modifier_variance, min=1e-12))

    return {
        "means": means.cpu(),
        "stds": stds.cpu(),
        "modifiers_mean": modifier_mean.cpu(),
        "modifiers_std": modifier_std.cpu(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute feature normalization statistics."
    )
    parser.add_argument(
        "--indices_path",
        type=str,
        default=str(repo_root / "generated/indices.parquet"),
        help="Full path to indices.parquet used to build the dataset.",
    )
    parser.add_argument(
        "--memmap_dir",
        type=str,
        default=str(repo_root / "generated/memmap"),
        help="Directory containing memmap arrays and idx_map_global.pkl.",
    )
    parser.add_argument(
        "--date_start",
        type=str,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--date_end",
        type=str,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Max sequence length passed to collate_auctions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit number of batches for a quick run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "generated/feature_stats.pt"),
        help="Full path for feature_stats.pt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    indices_path = Path(args.indices_path)
    memmap_dir = Path(args.memmap_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Using indices: %s", indices_path)
    logger.info("Using memmap dir: %s", memmap_dir)
    logger.info("Output path: %s", output_path)

    train_pairs = load_training_pairs(
        indices_path=indices_path,
        date_start=args.date_start,
        date_end=args.date_end,
    )

    idx_map_path = memmap_dir / "idx_map_global.pkl"
    logger.info("Loading global index map from %s", idx_map_path)
    with open(idx_map_path, "rb") as f:
        idx_map_global = pickle.load(f)

    dataset = AuctionDataset(
        pairs=train_pairs,
        idx_map_global=idx_map_global,
        feature_stats=None,
        root=str(memmap_dir),
    )

    batch_size = args.batch_size
    num_workers = args.num_workers

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_auctions(
            b, max_sequence_length=args.max_sequence_length
        ),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    logger.info(
        "Starting feature stats computation (batch_size=%d, workers=%d, max_batches=%s)",
        batch_size,
        num_workers,
        str(args.max_batches),
    )
    feature_stats = compute_feature_stats(dataloader, max_batches=args.max_batches)

    torch.save(feature_stats, output_path)
    logger.info("Saved feature stats to %s", output_path)

    logger.info("Auction feature means: %s", feature_stats["means"].tolist())
    logger.info("Auction feature stds: %s", feature_stats["stds"].tolist())
    logger.info(
        "Modifier mean/std: %f / %f",
        feature_stats["modifiers_mean"].item(),
        feature_stats["modifiers_std"].item(),
    )


if __name__ == "__main__":
    main()
