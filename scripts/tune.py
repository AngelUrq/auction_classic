import argparse
import logging
import sys
from pathlib import Path

import lightning as L
import optuna
import torch
import yaml
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from scripts.train import (
    create_dataloaders,
    create_model,
    load_config,
    load_data,
    load_mappings,
)


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for a single Optuna trial."""
    d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])

    valid_nheads = [n for n in [4, 8, 16] if d_model % n == 0]
    nhead = trial.suggest_categorical("nhead", valid_nheads)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": trial.suggest_int("num_layers", 2, 8),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [16, 32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "classification_loss_weight": trial.suggest_float(
            "classification_loss_weight", 10, 200, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def objective(
    trial: optuna.Trial,
    mappings: dict,
    feature_stats: dict,
    train_pairs,
    val_pairs,
    cfg,
    data_dir: Path,
    use_wandb: bool,
    study_name: str,
) -> tuple[float, float]:
    """Optuna objective: train one epoch and return (val/general_mae, val/classification_loss)."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    params = suggest_hyperparameters(trial)

    cfg.training.learning_rate = params["learning_rate"]
    cfg.training.batch_size = params["batch_size"]
    cfg.model.d_model = params["d_model"]
    cfg.model.nhead = params["nhead"]
    cfg.model.num_layers = params["num_layers"]
    cfg.model.dim_feedforward = params["dim_feedforward"]
    cfg.model.embedding_dim = params["embedding_dim"]
    cfg.model.dropout = params["dropout"]
    cfg.model.classification_loss_weight = params["classification_loss_weight"]

    logger.info(f"Trial {trial.number} params: {params}")

    model, param_count = create_model(mappings, cfg)

    train_dataloader, val_dataloader = create_dataloaders(
        train_pairs, val_pairs, feature_stats, cfg, data_dir
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.project,
            name=f"{study_name}-trial-{trial.number}",
            tags=["optuna", study_name],
        )
        wandb_logger.log_hyperparams(params)

    trainer = L.Trainer(
        max_epochs=1,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        log_every_n_steps=cfg.training.log_every,
        logger=wandb_logger,
        limit_val_batches=cfg.training.limit_val_batches,
        val_check_interval=cfg.training.val_check_interval,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip,
        enable_checkpointing=False,
    )

    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"Trial {trial.number} OOM: {e}")
            torch.cuda.empty_cache()
            if wandb_logger is not None:
                import wandb

                wandb.finish(exit_code=1)
            raise optuna.TrialPruned(f"OOM with params: {params}")
        raise

    metrics = trainer.callback_metrics
    general_mae = metrics.get("val/general_mae_epoch", metrics.get("val/general_mae"))
    classification_loss = metrics.get(
        "val/classification_loss", metrics.get("val/classification_loss_epoch")
    )

    if general_mae is None or classification_loss is None:
        raise optuna.TrialPruned(
            f"Missing metrics: general_mae={general_mae}, classification_loss={classification_loss}"
        )

    general_mae = float(general_mae)
    classification_loss = float(classification_loss)

    logger.info(
        f"Trial {trial.number} finished: "
        f"general_mae={general_mae:.4f}, classification_loss={classification_loss:.4f}"
    )

    if wandb_logger is not None:
        import wandb

        wandb.finish()

    return general_mae, classification_loss


def show_results(study: optuna.Study) -> None:
    """Print Pareto front trials from a multi-objective study."""
    pareto_trials = study.best_trials

    if not pareto_trials:
        logger.info("No completed trials found.")
        return

    logger.info(f"\nPareto front ({len(pareto_trials)} trials):")
    logger.info("=" * 80)

    for trial in sorted(pareto_trials, key=lambda t: t.values[0]):
        logger.info(
            f"  Trial {trial.number}: "
            f"general_mae={trial.values[0]:.4f}, "
            f"classification_loss={trial.values[1]:.4f}"
        )
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        logger.info("")


def save_pareto_front(study: optuna.Study, output_path: Path) -> None:
    """Save Pareto front trial parameters to a YAML file."""
    pareto_trials = study.best_trials

    if not pareto_trials:
        logger.info("No completed trials to save.")
        return

    results = []
    for trial in sorted(pareto_trials, key=lambda t: t.values[0]):
        results.append(
            {
                "trial_number": trial.number,
                "values": {
                    "general_mae": trial.values[0],
                    "classification_loss": trial.values[1],
                },
                "params": dict(trial.params),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Pareto front saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the tuning script."""
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Number of trials to run"
    )
    parser.add_argument(
        "--study-name", type=str, default="auction-sweep", help="Optuna study name"
    )
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Show results from an existing study and exit",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "configs" / "transformer.yaml"),
        help="Path to config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    studies_dir = repo_root / "optuna_studies"
    studies_dir.mkdir(exist_ok=True)
    storage = f"sqlite:///{studies_dir / args.study_name}.db"

    if args.show_results:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        show_results(study)
        pareto_path = studies_dir / f"{args.study_name}_pareto.yaml"
        save_pareto_front(study, pareto_path)
        return

    cfg = load_config(Path(args.config))

    data_dir = repo_root / cfg.data.dir

    logger.info("Loading mappings...")
    mappings = load_mappings(data_dir)

    logger.info("Loading feature statistics...")
    feature_stats = torch.load(data_dir / "feature_stats.pt", weights_only=False)

    logger.info("Loading data...")
    train_pairs, val_pairs = load_data(cfg, data_dir)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )

    use_wandb = not args.no_wandb and cfg.logging.wandb

    study.optimize(
        lambda trial: objective(
            trial,
            mappings,
            feature_stats,
            train_pairs,
            val_pairs,
            cfg,
            data_dir,
            use_wandb,
            args.study_name,
        ),
        n_trials=args.n_trials,
    )

    show_results(study)

    pareto_path = studies_dir / f"{args.study_name}_pareto.yaml"
    save_pareto_front(study, pareto_path)


if __name__ == "__main__":
    main()
