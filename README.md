# WoW Auction House Price Predictor

A machine learning system for World of Warcraft auction house analysis and price prediction. This project collects real-time auction data from the Blizzard API, trains deep learning models to predict item sale times, and provides actionable trading recommendations through an interactive UI.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Inference & UI](#inference--ui)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Overview

This system predicts how long items will remain on the auction house before selling, enabling profitable trading strategies. It uses:

- **Transformer-based neural networks** for sequence modeling of auction history
- **Quantile regression** for uncertainty estimation (predicting 10th, 50th, and 90th percentiles)
- **Historical context** from up to 24 hours of past auction snapshots
- **Rich item features** including bonuses, modifiers, and temporal patterns

## Features

### Automated Data Collection
- Hourly auction house snapshots via Blizzard API
- Intelligent deduplication with hash-based change detection
- Organized storage by date (YYYY/MM/DD structure)

### Deep Learning Models
- **AuctionTransformer**: Multi-head attention with item/context/bonus embeddings
- **AuctionRNN**: Bidirectional GRU encoder-decoder architecture
- Quantile loss for probabilistic predictions

### Feature Engineering
- Competitor price analysis (median, min, max, rank)
- Relative price positioning
- Temporal features (hour of week, time on sale)
- Item metadata (quality, class, subclass, stackability)

### Interactive UI
- Real-time auction browser with filtering
- Automated flip recommendations
- Individual item prediction tool
- Profit/loss estimation

### Reinforcement Learning Environment
- Gymnasium-compatible auction trading environment
- Simulated buying/selling with realistic fees
- Bankruptcy penalties and profit rewards

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Blizzard API   │────▶│  Data Collection │────▶│  JSON Storage   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Gradio UI     │◀────│    Inference     │◀────│  HDF5 Sequences │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  PyTorch Model   │◀────│    Training     │
                        └──────────────────┘     └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- Blizzard Developer API credentials

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/auction_classic.git
   cd auction_classic
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API credentials**
   
   Update the following files with your Blizzard API credentials:
   - `scripts/data_collection/retrieve_auctions.sh`
   - `src/data/utils.py`

   ```bash
   client_key="YOUR_CLIENT_KEY"
   secret_key="YOUR_SECRET_KEY"
   realm_id="YOUR_REALM_ID"  # e.g., "3676" for Area 52
   ```

   Get your API credentials from the [Blizzard Developer Portal](https://develop.battle.net/).

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `lightning` | Deep learning framework |
| `pandas` / `numpy` | Data manipulation |
| `h5py` | Efficient sequence storage |
| `scikit-learn` | Preprocessing & evaluation |
| `wandb` | Experiment tracking |
| `gradio` | Web UI |
| `tqdm` | Progress bars |

## Data Collection

### Automated Hourly Collection

Set up a cron job to collect auction data every hour:

```bash
crontab -e
```

Add the following line:
```bash
0 * * * * /path/to/auction_classic/scripts/data_collection/retrieve_auctions.sh
```

### Manual Collection

```bash
cd scripts/data_collection
chmod +x retrieve_auctions.sh
./retrieve_auctions.sh
```

### Data Format

Raw auction data is stored as JSON files with the following structure:

```json
{
  "auctions": [
    {
      "id": 123456789,
      "item": {
        "id": 12345,
        "context": 5,
        "bonus_lists": [1, 2, 3],
        "modifiers": [{"type": 1, "value": 100}]
      },
      "bid": 10000000,
      "buyout": 15000000,
      "quantity": 1,
      "time_left": "VERY_LONG"
    }
  ]
}
```

### Time Left Values

| Value | Duration |
|-------|----------|
| `SHORT` | 0.5 hours |
| `MEDIUM` | 2 hours |
| `LONG` | 12 hours |
| `VERY_LONG` | 48 hours |

## Data Processing

### 1. Compute Timestamps

Calculate first/last appearance for each auction:

```bash
python scripts/data_processing/compute_timestamps.py \
    --data_dir data/tww/auctions/ \
    --output generated/timestamps.json
```

### 2. Process Mappings

Generate vocabulary mappings for categorical features:

```bash
python scripts/data_processing/process_mappings.py \
    --data_dir data/tww/auctions/ \
    --output_dir generated/mappings/
```

This creates:
- `item_to_idx.json` - Item ID → index mapping
- `context_to_idx.json` - Context ID → index mapping  
- `bonus_to_idx.json` - Bonus ID → index mapping
- `modtype_to_idx.json` - Modifier type → index mapping

### 3. Prepare Sequence Data

Convert raw JSON to optimized HDF5 format:

```bash
python scripts/data_processing/prepare_sequence_data.py \
    --data_dir data/tww/auctions/ \
    --timestamps generated/timestamps.json \
    --mappings_dir generated/mappings/ \
    --output_dir generated/
```

Output:
- `sequences.h5` - HDF5 file with per-item auction sequences
- `indices.parquet` - Index file mapping (item, timestamp) → HDF5 positions

### HDF5 Structure

```
sequences.h5
└── items/
    └── {item_index}/
        ├── data           # (N, 6) float32: bid, buyout, quantity, time_left, current_hours, hours_on_sale
        ├── contexts       # (N,) int32: context IDs
        ├── bonus_lists    # (N, 9) int32: bonus IDs (padded)
        ├── modifier_types # (N, 11) int32: modifier type IDs (padded)
        └── modifier_values# (N, 11) float32: modifier values (padded)
```

## Model Training

### Transformer Model

The primary model uses a Transformer encoder with learned embeddings:

```bash
python -c "
from src.models.auction_transformer import AuctionTransformer
import lightning as L

model = AuctionTransformer(
    input_size=5,
    n_items=50000,
    n_contexts=100,
    n_bonuses=5000,
    n_modtypes=50,
    embedding_dim=128,
    d_model=512,
    nhead=4,
    num_layers=4,
    max_hours_back=24,
    quantiles=[0.1, 0.5, 0.9]
)

trainer = L.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, train_dataloader, val_dataloader)
"
```

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Transformer hidden dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers` | 4 | Number of transformer layers |
| `embedding_dim` | 128 | Embedding dimension for categorical features |
| `dropout_p` | 0.1 | Dropout probability |
| `learning_rate` | 1e-4 | Initial learning rate |
| `max_hours_back` | 24 | Historical context window (hours) |

### Training Features

- **Quantile Loss**: Predicts 10th, 50th, and 90th percentiles for uncertainty estimation
- **OneCycleLR Scheduler**: Cosine annealing with warmup
- **Weighted Loss**: Exponential decay weighting based on `current_hours`
- **W&B Integration**: Automatic logging of metrics, gradients, and predictions

### RNN Alternative

For comparison, an RNN-based model is also available:

```python
from src.models.auction_rnn import AuctionRNN

model = AuctionRNN(
    n_items=50000,
    input_size=5,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
    num_layers=2,
    bidirectional=True
)
```

## Inference & UI

### Launch the Web UI

```bash
cd ui
python app.py
```

This starts a Gradio interface with three tabs:

### 1. Current Auctions Tab
- Browse live auction data
- Filter by item ID
- View historical snapshots (time offset slider)
- Generate predictions for displayed items

### 2. Recommendations Tab
- Automated flip recommendations
- Configurable profit threshold
- Sale time filtering (e.g., items predicted to sell within 4 hours)
- Sorted by potential profit

### 3. Individual Flipping Tab
- Search for specific items
- Set custom resale prices
- Get predicted sale times for your listing

### Programmatic Inference

```python
import torch
from src.models.auction_transformer import AuctionTransformer
from src.models.inference import predict_dataframe

# Load model
model = AuctionTransformer.load_from_checkpoint('models/checkpoint.ckpt')
model.eval()

# Load feature statistics
feature_stats = torch.load('generated/feature_stats.pt')

# Predict
predictions = predict_dataframe(
    model=model,
    df_auctions=df,
    prediction_time=datetime.now(),
    feature_stats=feature_stats,
    max_hours_back=24
)
```

## Project Structure

```
auction_classic/
├── data/                          # Static item data
│   ├── items.csv                  # Item metadata
│   ├── items_wotlk.csv           # WotLK Classic items
│   └── items_cata.csv            # Cataclysm Classic items
├── notebooks/                     # Jupyter notebooks
│   ├── eda.ipynb                 # Exploratory data analysis
│   ├── feature_normalization.ipynb
│   ├── forest.ipynb              # Random Forest baseline
│   ├── forest_evaluation.ipynb
│   ├── transformer_training.ipynb
│   └── transformer_evaluation.ipynb
├── scripts/
│   ├── analysis/                  # Analysis scripts
│   │   ├── benchmark_dataloader.py
│   │   ├── historical.py
│   │   └── weekly_hours.py
│   ├── data_collection/           # Data retrieval
│   │   ├── retrieve_auctions.sh
│   │   ├── retrieve_commodities.sh
│   │   ├── retrieve_items.py
│   │   └── sync_auctions.sh
│   ├── data_processing/           # Data pipeline
│   │   ├── compute_timestamps.py
│   │   ├── find_max_bonuses_modifiers.py
│   │   ├── merge_items.py
│   │   ├── prepare_sequence_data.py
│   │   └── process_mappings.py
│   └── validation/                # Data validation
│       ├── validate_data.py
│       └── verify_daily_data.sh
├── src/
│   ├── data/
│   │   ├── auction_dataset.py    # PyTorch Dataset
│   │   ├── auction_preprocessor.py # Feature engineering
│   │   └── utils.py              # Data utilities
│   ├── models/
│   │   ├── auction_transformer.py # Main model
│   │   ├── auction_rnn.py        # RNN baseline
│   │   └── inference.py          # Prediction utilities
│   ├── rl/
│   │   └── auction_env.py        # Gymnasium environment
│   ├── sql/
│   │   ├── auction_tables.sql    # Database schema
│   │   └── auction_queries.sql   # Useful queries
│   └── training/
│       └── train.py              # Training loop
├── ui/
│   └── app.py                    # Gradio web interface
├── requirements.txt
└── README.md
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | Weights & Biases API key |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

### Key Paths

| Path | Description |
|------|-------------|
| `data/tww/auctions/` | Raw auction JSON files |
| `generated/` | Processed data and model artifacts |
| `generated/mappings/` | Vocabulary mapping files |
| `models/` | Trained model checkpoints |

## Database Schema

For SQL-based storage (optional):

```sql
-- Items table
CREATE TABLE Items (
    item_id INT PRIMARY KEY,
    item_name VARCHAR(100),
    quality VARCHAR(50),
    item_level INT,
    item_class VARCHAR(50),
    item_subclass VARCHAR(50)
);

-- Auctions table
CREATE TABLE Auctions (
    auction_id INT PRIMARY KEY,
    bid FLOAT,
    buyout FLOAT,
    quantity INT,
    item_id INT REFERENCES Items(item_id)
);

-- Auction events (for tracking over time)
CREATE TABLE AuctionEvents (
    auction_id INT,
    record DATETIME,
    time_left VARCHAR(20),
    PRIMARY KEY (auction_id, record)
);
```

## Metrics & Evaluation

### Training Metrics

- **Pinball Loss**: Quantile regression loss
- **MAE (hours)**: Mean absolute error in predicted hours
- **Coverage**: % of targets within predicted intervals
- **Calibration**: Observed vs. expected quantile fractions

### Business Metrics

- **Potential Profit**: Target price - Purchase price
- **Sale Probability**: Based on predicted hours to sale
- **ROI**: Return on investment for recommended flips

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is for educational purposes. World of Warcraft and Blizzard Entertainment are trademarks of Blizzard Entertainment, Inc.

## Acknowledgments

- [Blizzard API](https://develop.battle.net/) for auction house data access
- [PyTorch Lightning](https://lightning.ai/) for training infrastructure
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [Gradio](https://gradio.app/) for the web interface
