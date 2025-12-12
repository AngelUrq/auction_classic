import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from functools import partial
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from src.data.auction_dataset import AuctionDataset
from src.data.bucket_sampler import BucketBatchSampler, get_lengths
from src.data.utils import collate_auctions
from src.models.auction_transformer import AuctionTransformer


def format_param_count(count: int) -> str:
    """Format parameter count as human readable string (e.g., 4M, 1.2M, 500K)."""
    if count >= 1_000_000:
        if count % 1_000_000 == 0:
            return f"{count // 1_000_000}M"
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        if count % 1_000 == 0:
            return f"{count // 1_000}K"
        return f"{count / 1_000:.1f}K"
    return str(count)


def load_mappings(data_dir: Path) -> dict:
    """Load all mapping files from the mappings directory."""
    mappings_dir = data_dir / "mappings"
    mappings = {}

    mapping_files = [
        "item_to_idx.json",
        "context_to_idx.json",
        "bonus_to_idx.json",
        "modtype_to_idx.json",
    ]

    for filename in mapping_files:
        filepath = mappings_dir / filename
        with open(filepath, "r") as f:
            key = filename.replace(".json", "").replace("_to_idx", "")
            mappings[key] = json.load(f)
        logger.info(f"Loaded {filename}: {len(mappings[key])} entries")

    return mappings


def load_data(cfg: DictConfig, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split the training data."""
    filters = [
        ("record", ">=", cfg.data.date_start),
        ("record", "<=", cfg.data.date_end),
    ]

    logger.info(f"Loading data from {data_dir / 'indices.parquet'}")
    logger.info(f"  Date range: {cfg.data.date_start} to {cfg.data.date_end}")

    pairs = pd.read_parquet(
        data_dir / "indices.parquet",
        engine="pyarrow",
        filters=filters
    )

    logger.info(f"  Total pairs: {len(pairs):,}")

    split_idx = int(len(pairs) * cfg.data.train_split)
    train_val_pairs = pairs.iloc[:split_idx]
    train_pairs = train_val_pairs.iloc[:int(len(train_val_pairs) * cfg.data.train_fraction)]
    val_pairs = pairs.iloc[split_idx:]

    logger.info(f"  Train pairs: {len(train_pairs):,}")
    logger.info(f"  Val pairs: {len(val_pairs):,}")

    return train_pairs, val_pairs


def create_dataloaders(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    feature_stats: dict,
    cfg: DictConfig,
    data_dir: Path,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation dataloaders."""
    train_pairs = train_pairs[["record", "item_index"]]
    val_pairs = val_pairs[["record", "item_index"]]

    # Load global index map from pickle
    memmap_root = str(data_dir / "memmap")
    idx_map_path = data_dir / "memmap" / "idx_map_global.pkl"
    logger.info(f"Loading global index map from {idx_map_path}")
    with open(idx_map_path, "rb") as f:
        global_idx_map = pickle.load(f)
    logger.info(f"  Loaded {len(global_idx_map):,} entries")

    train_dataset = AuctionDataset(
        pairs=train_pairs,
        idx_map_global=global_idx_map,
        feature_stats=feature_stats,
        root=memmap_root,
        max_hours_back=cfg.data.max_hours_back,
    )
    val_dataset = AuctionDataset(
        pairs=val_pairs,
        idx_map_global=global_idx_map,
        feature_stats=feature_stats,
        root=memmap_root,
        max_hours_back=cfg.data.max_hours_back,
    )

    num_workers = cfg.training.num_workers
    prefetch = cfg.data.prefetch_factor if num_workers > 0 else None
    collate_fn = partial(
        collate_auctions,
        max_sequence_length=int(cfg.data.max_sequence_length),
    )

    if cfg.data.bucket_sampling:
        logger.info(f"  Using bucket sampling with {cfg.data.num_buckets} buckets")
        logger.info("  Loading/computing sequence lengths...")
        train_lengths = get_lengths(
            train_pairs, global_idx_map, cfg.data.max_hours_back, cache_dir=str(data_dir), split="train"
        )
        val_lengths = get_lengths(
            val_pairs, global_idx_map, cfg.data.max_hours_back, cache_dir=str(data_dir), split="val"
        )

        train_sampler = BucketBatchSampler(
            train_lengths,
            batch_size=cfg.training.batch_size,
            num_buckets=cfg.data.num_buckets,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = BucketBatchSampler(
            val_lengths,
            batch_size=cfg.training.batch_size,
            num_buckets=cfg.data.num_buckets,
            shuffle=False,
            drop_last=False,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
    else:
        logger.info("  Using random sampling")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )

    return train_dataloader, val_dataloader


def create_model(mappings: dict, cfg: DictConfig) -> tuple[AuctionTransformer, int]:
    """Create the AuctionTransformer model and return it with parameter count."""
    n_items = len(mappings["item"])
    n_contexts = len(mappings["context"]) + 1
    n_bonuses = len(mappings["bonus"])
    n_modtypes = len(mappings["modtype"])

    model = AuctionTransformer(
        input_size=cfg.model.input_size,
        n_items=n_items,
        n_contexts=n_contexts,
        n_bonuses=n_bonuses,
        n_modtypes=n_modtypes,
        embedding_dim=cfg.model.embedding_dim,
        d_model=cfg.model.d_model,
        dim_feedforward=cfg.model.dim_feedforward,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout_p=float(cfg.model.dropout),
        learning_rate=float(cfg.training.learning_rate),
        logging_interval=int(cfg.training.logging_interval),
        quantiles=list(cfg.model.quantiles),
        max_hours_back=int(cfg.data.max_hours_back),
        log_raw_batch_data=False,
        log_step_predictions=False,
    )

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    return model, param_count


def generate_run_name(param_count: int, max_hours_back: int) -> str:
    """Generate run name based on parameters and config."""
    param_str = format_param_count(param_count)
    return f"transformer-{param_str}-quantile-historical_{max_hours_back}"


def load_config(config_path: Path) -> DictConfig:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return OmegaConf.create(config_dict)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Auction Transformer model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "configs" / "transformer.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = load_config(Path(args.config))
    
    # Override resume from command line if provided
    if args.resume:
        cfg.resume = args.resume
    
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    data_dir = repo_root / cfg.data.dir
    checkpoint_dir = repo_root / cfg.checkpoint.dir
    checkpoint_dir.mkdir(exist_ok=True)

    logger.info("Loading mappings...")
    mappings = load_mappings(data_dir)

    logger.info("Creating model...")
    if cfg.resume:
        logger.info(f"Resuming from checkpoint: {cfg.resume}")
        model = AuctionTransformer.load_from_checkpoint(cfg.resume)
        param_count = sum(p.numel() for p in model.parameters())
    else:
        model, param_count = create_model(mappings, cfg)

    run_name = generate_run_name(param_count, cfg.data.max_hours_back)

    logger.info("=" * 60)
    logger.info("Auction Transformer Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {run_name}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Learning rate: {cfg.training.learning_rate}")
    logger.info(f"Max hours back: {cfg.data.max_hours_back}")
    logger.info(f"Bucket sampling: {cfg.data.bucket_sampling}")
    logger.info(f"Model: d_model={cfg.model.d_model}, layers={cfg.model.num_layers}, heads={cfg.model.nhead}")

    logger.info("Loading feature statistics...")
    feature_stats = torch.load(data_dir / "feature_stats.pt", weights_only=False)

    logger.info("Loading data...")
    train_pairs, val_pairs = load_data(cfg, data_dir)

    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_pairs, val_pairs, feature_stats, cfg, data_dir
    )

    del train_pairs, val_pairs

    wandb_logger = None
    if cfg.logging.wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.project,
            name=run_name,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir / run_name),
        filename="epoch_{epoch:02d}",
        save_top_k=cfg.checkpoint.save_top_k,
        every_n_train_steps=cfg.training.checkpoint_every,
        save_last=cfg.checkpoint.save_last,
    )

    logger.info("Initializing trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        log_every_n_steps=cfg.training.log_every,
        logger=wandb_logger,
        limit_val_batches=cfg.training.limit_val_batches,
        val_check_interval=cfg.training.val_check_interval,
        precision=cfg.training.precision,
        callbacks=[checkpoint_callback],
        gradient_clip_val=cfg.training.gradient_clip,
    )

    logger.info("=" * 60)
    logger.info("Validating model...")
    
    trainer.validate(
        model=model,
        dataloaders=val_dataloader,
    )

    logger.info("Starting training...")
    logger.info("=" * 60)

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=cfg.resume if cfg.resume else None,
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
