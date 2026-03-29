# WoW Auction House Sale Duration Predictor

A machine learning system for World of Warcraft auction house analysis. Collects real-time auction data from the Blizzard API, trains a Transformer model to predict how long items will take to sell, and provides trading recommendations through an interactive UI.

## Architecture

```
[Collector device]               [Database server]               [Client]
Blizzard API (hourly)       →    /mnt/archive/data/              UI + inference
  retrieve_auctions.sh           PostgreSQL
  retrieve_commodities.sh   →    ingest.py (cron)
  rsync → server                 fetch_items.py (cron)
  delete local files
```

## Setup

### Prerequisites

- Python 3.10+
- Docker + Docker Compose (on database server)
- Blizzard Developer API credentials ([Blizzard Developer Portal](https://develop.battle.net/))

### 1. Configure environment

Copy `.env.example` to `.env` and fill in all values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `BLIZZARD_CLIENT_KEY` | Blizzard API client key |
| `BLIZZARD_SECRET_KEY` | Blizzard API secret key |
| `BLIZZARD_REALM_ID` | Connected realm ID (e.g. 3676) |
| `BLIZZARD_REGION` | API region (e.g. `us`) |
| `BLIZZARD_NAMESPACE` | API namespace (e.g. `dynamic-us`) |
| `BLIZZARD_LOCALE` | Locale (e.g. `en_US`) |
| `SERVER_USER` | SSH user for the database server |
| `SERVER_HOST` | Database server IP or hostname |
| `SERVER_AUCTIONS_DIR` | Path on server for raw auction JSONs |
| `SERVER_COMMODITIES_DIR` | Path on server for raw commodity JSONs |
| `COLLECTOR_AUCTIONS_DIR` | Local path on the collector device for auctions |
| `COLLECTOR_COMMODITIES_DIR` | Local path on the collector device for commodities |
| `POSTGRES_URL` | PostgreSQL connection URL |
| `POSTGRES_USER` | PostgreSQL user (used by Docker) |
| `POSTGRES_PASSWORD` | PostgreSQL password (used by Docker) |
| `POSTGRES_DB` | PostgreSQL database name (used by Docker) |

### 2. Start PostgreSQL

```bash
docker compose up -d
```

### 3. Install dependencies

```bash
uv sync
# or to add a new dependency:
uv add <package>
```

## Cron Setup

### Collector device

Fetches auction data hourly from the Blizzard API, syncs to the server, and deletes local files:

```cron
0 * * * * cd /path/to/auction_classic && bash scripts/collect/retrieve_auctions.sh
0 * * * * cd /path/to/auction_classic && bash scripts/collect/retrieve_commodities.sh
```

### Database server

Ingests new JSON files into PostgreSQL and fetches missing item metadata:

```cron
# Ingest new auction/commodity JSONs (5 min offset to allow rsync to complete)
5 * * * * cd /path/to/auction_classic && python scripts/db/ingest.py

# Fetch up to 100 missing items from Blizzard API per run
30 * * * * cd /path/to/auction_classic && python scripts/db/fetch_items.py
```

## Data Collection Scripts

| Script | Description |
|--------|-------------|
| `scripts/collect/retrieve_auctions.sh` | Fetches hourly auction snapshots, rsyncs to server |
| `scripts/collect/retrieve_commodities.sh` | Fetches hourly commodity snapshots, rsyncs to server |
| `scripts/db/ingest.py` | Parses JSONs and upserts into PostgreSQL |
| `scripts/db/fetch_items.py` | Fetches missing item metadata from Blizzard API |

## Database Schema

Managed by `scripts/db/init.sql`, applied automatically on first Docker startup.

```
items                  — item metadata from Blizzard item API
auctions               — one row per auction listing (static data)
auction_observations   — one row per (auction, snapshot) — tracks time_left over time
auction_bonus          — bonus IDs per auction
auction_modifiers      — modifier type/value per auction
commodities            — one row per commodity listing
commodity_observations — one row per (commodity, snapshot) — tracks quantity and time_left
processed_files        — tracks which JSON files have been ingested
```

## Data Pipeline (Training)

```bash
# Run all transform steps
bash scripts/prepare_data.sh

# Or individually:
python scripts/transform/compute_timestamps.py --data_dir data/auctions/ --output_file generated/timestamps.json
python scripts/transform/process_mappings.py --data_dir data/auctions/ --output_dir generated/mappings/
python scripts/transform/prepare_sequence_data.py --data_dir data/auctions/ --timestamps generated/timestamps.json --mappings_dir generated/mappings/ --output_dir generated/
python scripts/transform/convert_hdf5_to_npy.py --h5_path generated/sequences.h5 --indices_path generated/indices.parquet --output_dir generated/memmap/
python scripts/transform/compute_feature_stats.py --indices_path generated/indices.parquet --memmap_dir generated/memmap/
```

## Model Training

```bash
python scripts/train.py
# Override config values:
python scripts/train.py training.batch_size=128 model.num_layers=6
```

Training is configured via Hydra (`configs/transformer.yaml`). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Transformer hidden dimension |
| `nhead` | 16 | Number of attention heads |
| `num_layers` | 4 | Transformer layers |
| `n_time_bins` | 48 | Discrete survival time bins |
| `data.max_hours_back` | 72 | Historical context window (hours) |

## UI

```bash
cd ui && python app.py
# Opens Gradio at http://localhost:7860
```

Three tabs:
1. **Current Auctions** — browse live auctions with time-offset slider
2. **Recommendations** — automated flip suggestions with profit threshold
3. **Individual Flipping** — search specific items with custom resale price

## Validation

```bash
python scripts/validate/validate_data.py
bash scripts/validate/verify_daily_data.sh
```
