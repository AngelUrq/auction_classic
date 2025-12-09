import json
import sys
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from src.data.auction_dataset import AuctionDataset
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
        print(f"Loaded {filename}: {len(mappings[key])} entries")

    return mappings


def load_data(cfg: DictConfig, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split the training data."""
    filters = [
        ("record", ">=", cfg.data.date_start),
        ("record", "<=", cfg.data.date_end),
    ]

    print(f"Loading data from {data_dir / 'indices.parquet'}")
    print(f"  Date range: {cfg.data.date_start} to {cfg.data.date_end}")

    pairs = pd.read_parquet(
        data_dir / "indices.parquet",
        engine="pyarrow",
        filters=filters
    )

    print(f"  Total pairs: {len(pairs):,}")

    split_idx = int(len(pairs) * cfg.data.train_split)
    train_val_pairs = pairs.iloc[:split_idx]
    train_pairs = train_val_pairs.iloc[:int(len(train_val_pairs) * cfg.data.train_fraction)]
    val_pairs = pairs.iloc[split_idx:]

    print(f"  Train pairs: {len(train_pairs):,}")
    print(f"  Val pairs: {len(val_pairs):,}")

    return train_pairs, val_pairs


def create_dataloaders(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    feature_stats: dict,
    cfg: DictConfig,
    data_dir: Path,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation dataloaders."""
    train_pairs = train_pairs[["record", "item_index", "start", "length"]]
    val_pairs = val_pairs[["record", "item_index", "start", "length"]]

    train_idx_map = {
        (int(row.item_index), row.record): (int(row.start), int(row.length))
        for row in train_pairs.itertuples(index=False)
    }
    val_idx_map = {
        (int(row.item_index), row.record): (int(row.start), int(row.length))
        for row in val_pairs.itertuples(index=False)
    }

    h5_path = str(data_dir / "sequences.h5")
    train_dataset = AuctionDataset(
        train_pairs,
        train_idx_map,
        feature_stats=feature_stats,
        max_hours_back=cfg.data.max_hours_back,
        path=h5_path,
    )
    val_dataset = AuctionDataset(
        val_pairs,
        val_idx_map,
        feature_stats=feature_stats,
        max_hours_back=cfg.data.max_hours_back,
        path=h5_path,
    )

    num_workers = cfg.training.num_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_auctions,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_auctions,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
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
        dropout_p=cfg.model.dropout,
        learning_rate=cfg.training.learning_rate,
        logging_interval=cfg.training.logging_interval,
        quantiles=list(cfg.model.quantiles),
        max_hours_back=cfg.data.max_hours_back,
        log_raw_batch_data=False,
        log_step_predictions=False,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    return model, param_count


def generate_run_name(param_count: int, max_hours_back: int) -> str:
    """Generate run name based on parameters and config."""
    param_str = format_param_count(param_count)
    return f"transformer-{param_str}-quantile-historical_{max_hours_back}"


@hydra.main(version_base=None, config_path="../configs", config_name="transformer")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    data_dir = repo_root / cfg.data.dir
    checkpoint_dir = repo_root / cfg.checkpoint.dir
    checkpoint_dir.mkdir(exist_ok=True)

    print("\nLoading mappings...")
    mappings = load_mappings(data_dir)

    print("\nCreating model...")
    if cfg.resume:
        print(f"Resuming from checkpoint: {cfg.resume}")
        model = AuctionTransformer.load_from_checkpoint(cfg.resume)
        param_count = sum(p.numel() for p in model.parameters())
    else:
        model, param_count = create_model(mappings, cfg)

    run_name = generate_run_name(param_count, cfg.data.max_hours_back)

    print("=" * 60)
    print("Auction Transformer Training")
    print("=" * 60)
    print(f"\nExperiment: {run_name}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Max hours back: {cfg.data.max_hours_back}")
    print(f"Model: d_model={cfg.model.d_model}, layers={cfg.model.num_layers}, heads={cfg.model.nhead}")

    print("\nLoading feature statistics...")
    feature_stats = torch.load(data_dir / "feature_stats.pt", weights_only=False)

    print("\nLoading data...")
    train_pairs, val_pairs = load_data(cfg, data_dir)

    print("\nCreating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_pairs, val_pairs, feature_stats, cfg, data_dir
    )

    del train_pairs, val_pairs

    logger = None
    if cfg.logging.wandb:
        logger = WandbLogger(
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

    print("\nInitializing trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        log_every_n_steps=cfg.training.log_every,
        logger=logger,
        limit_val_batches=cfg.training.limit_val_batches,
        val_check_interval=cfg.training.val_check_interval,
        precision=cfg.training.precision,
        callbacks=[checkpoint_callback],
        gradient_clip_val=cfg.training.gradient_clip,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=cfg.resume if cfg.resume else None,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
