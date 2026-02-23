# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WoW auction house price predictor: collects Blizzard API data, trains a Transformer model to predict item sale duration (quantile regression + binary classification), and serves results via a Gradio web UI.

## Common Commands

**Install dependencies:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run training** (Hydra config at `configs/transformer.yaml`):
```bash
python scripts/train.py
```

**Override Hydra config values:**
```bash
python scripts/train.py training.batch_size=128 model.num_layers=6
```

**Run UI:**
```bash
cd ui && python app.py
# Opens Gradio at http://localhost:7860
```

**Data pipeline** (or just run `bash scripts/prepare_data.sh`):
```bash
python scripts/transform/compute_timestamps.py --data_dir data/auctions/ --output_file generated/timestamps.json
python scripts/transform/process_mappings.py --data_dir data/auctions/ --output_dir generated/mappings/
python scripts/transform/prepare_sequence_data.py --data_dir data/auctions/ --timestamps generated/timestamps.json --mappings_dir generated/mappings/ --output_dir generated/
python scripts/transform/convert_hdf5_to_npy.py --h5_path generated/sequences.h5 --indices_path generated/indices.parquet --output_dir generated/memmap/
python scripts/transform/compute_feature_stats.py --indices_path generated/indices.parquet --memmap_dir generated/memmap/
```

**Validate data:**
```bash
python scripts/validate/validate_data.py
bash scripts/validate/verify_daily_data.sh
```

**Benchmark dataloader:**
```bash
python scripts/analyze/benchmark_dataloader.py
```

## Architecture

### Data Flow
```
[Blizzard API hourly cron] → data/auctions/YYYY/MM/DD/HHZ.json
    → compute_timestamps.py  → generated/timestamps.json
    → process_mappings.py    → generated/mappings/{item,context,bonus,modtype}_to_idx.json
    → prepare_sequence_data.py → generated/sequences.h5 + generated/indices.parquet
    → convert_hdf5_to_npy.py → generated/memmap/*.npy
    → compute_feature_stats.py → generated/feature_stats.pt
    → AuctionDataset (PyTorch) → AuctionTransformer → predictions
    → ui/app.py (Gradio)
```

### Core Model (`src/models/auction_transformer.py`)
Multi-task Transformer (~4.2M params) built with PyTorch Lightning:

**Inputs per auction record:**
- `auction_features`: (B, S, 5) — bid, buyout, quantity, time_left, listing_age
- `item_index`, `contexts`: categorical IDs → learned embeddings
- `bonus_ids`: (B, S, 9) — bonus IDs, conditioned on (item, context) via FiLM
- `modifier_types`, `modifier_values`: (B, S, 11) — modifier embeddings + projected scalars
- `hour_of_week` (0–167), `snapshot_offset` (0–72): temporal embeddings

**Architecture:**
- FiLM conditioning: context modulates item embeddings
- Bonus/modifier aggregation with mask-aware averaging
- TransformerEncoder: 4 layers, d_model=64, nhead=4, dim_feedforward=256
- **Regression head**: → 3 quantile outputs (τ=0.1, 0.5, 0.9) via pinball loss
- **Classification head**: → 1 binary logit (is_short_duration, threshold=8h) via BCE

**Training details:**
- AdamW + OneCycleLR scheduler
- Exponential sample weighting (recent listings weighted higher)
- Gradient clipping: 3.0, precision: bfloat16
- Experiment tracking: Weights & Biases

### Data Loading (`src/data/`)
- `AuctionDataset`: Memory-mapped numpy backend (`generated/memmap/*.npy`) with per-item sequence loading
- `BucketBatchSampler`: Groups by sequence length (20 buckets) to reduce padding waste
- `collate_auctions()`: Crops from left (most recent), right-pads to batch max length

### Inference (`src/models/inference.py`)
`predict_dataframe()` takes live auction DataFrame → groups by item → batch inference → returns q10/q50/q90 + `is_short_duration` probability.

### UI (`ui/app.py`)
Three Gradio tabs:
1. **Current Auctions**: Browse live auctions with time-offset slider
2. **Recommendations**: Automated flip suggestions with profit threshold
3. **Individual Flipping**: Search specific items with custom resale price

### Configuration (`configs/transformer.yaml`)
Key tunable parameters:
- `data.date_start` / `data.date_end`: Training date range
- `data.max_hours_back`: Historical context window (default 72h)
- `model.pretrained_path`: Checkpoint to fine-tune from
- `model.reset_embeddings`: Reinitialize embeddings when loading checkpoint
- `training.classification_loss_weight`: Balance between quantile and classification tasks

### Generated Artifacts
- `generated/mappings/`: Vocabulary JSONs (17,955 items, 64 contexts, ~1,405 bonuses)
- `generated/indices.parquet`: Maps (item_index, record) → HDF5 position
- `generated/feature_stats.pt`: Precomputed means/stds for normalization
- `models/`: Lightning checkpoints (`.ckpt`)

### Alternative Model
`src/models/auction_rnn.py`: Bidirectional GRU baseline (hidden_size=128) used for comparison.

### RL Component
`src/rl/auction_env.py`: Gymnasium-compatible environment for simulated trading experiments.

## Code Style

- Every test function must have a docstring explaining what behaviour it verifies.
- Helper function names must be descriptive actions: prefer `_make_X`, `_build_X`, `_write_X` over abbreviations (`_pa_args`, `_pm_args`) or vague nouns (`_simple_dataset`, `_setup_X`).
- Variable and attribute names must use full, descriptive words — never abbreviate (e.g. `_val_survival_curves` not `_val_surv_funcs`).
- Keep step methods (`training_step`, `validation_step`, etc.) concise: extract logic into well-named private helpers (`_compute_X`, `_build_X`, `_accumulate_X`, `_clear_X`) rather than inlining multi-step operations.
