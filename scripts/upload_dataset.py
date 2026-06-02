"""Upload training dataset to HuggingFace Hub.

Uploads the subset needed for training (memmap/, cache/, indices.parquet,
feature_stats.pt, mappings/) and skips large intermediates.

Usage:
    python scripts/upload_dataset.py
    python scripts/upload_dataset.py --repo your-username/auction-data
    python scripts/upload_dataset.py --generated_dir /path/to/generated
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UPLOAD_PATHS = [
    "memmap",
    "cache",
    "indices.parquet",
    "feature_stats.pt",
    "mappings",
]

DATASET_CARD = """\
---
language:
- en
tags:
- wow
- auction-house
- time-series
license: mit
---

# WoW Auction House Dataset

Historical World of Warcraft auction house listings with item metadata, pricing,
and sale outcome labels.

## Contents

| Path | Description |
|------|-------------|
| `memmap/` | Memory-mapped numpy arrays — main training data |
| `cache/` | Precomputed sequence cache |
| `indices.parquet` | Maps (item_index, record) → memmap position |
| `feature_stats.pt` | Per-feature means and stds for normalization |
| `mappings/` | Vocabulary JSONs (items, contexts, bonuses, modifier types) |

## Usage

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AngelZeur/auction-data",
    repo_type="dataset",
    local_dir="generated/",
)
```
"""


def _upload_dataset_card(api: HfApi, repo_id: str) -> None:
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
    )
    log.info("Uploaded dataset card")


def upload_dataset(repo_id: str, generated_dir: Path, private: bool) -> None:
    api = HfApi()

    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    log.info("Repo %s ready", repo_id)

    _upload_dataset_card(api, repo_id)

    for relative_path in UPLOAD_PATHS:
        local_path = generated_dir / relative_path
        if not local_path.exists():
            log.warning("Skipping %s — not found", local_path)
            continue

        if local_path.is_dir():
            log.info("Uploading folder %s …", relative_path)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(local_path),
                path_in_repo=relative_path,
            )
        else:
            log.info("Uploading file %s …", relative_path)
            api.upload_file(
                repo_id=repo_id,
                repo_type="dataset",
                path_or_fileobj=str(local_path),
                path_in_repo=relative_path,
            )

    log.info("Done. Download on VastAI with:")
    log.info(
        "  from huggingface_hub import snapshot_download\n"
        '  snapshot_download(repo_id="%s", repo_type="dataset", local_dir="generated/")',
        repo_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload training dataset to HuggingFace Hub")
    parser.add_argument("--repo", default="AngelZeur/auction-data", help="HF repo id (default: AngelZeur/auction-data)")
    parser.add_argument(
        "--generated_dir",
        default="generated",
        help="Path to the generated/ directory (default: generated/)",
    )
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    args = parser.parse_args()

    upload_dataset(
        repo_id=args.repo,
        generated_dir=Path(args.generated_dir),
        private=args.private,
    )


if __name__ == "__main__":
    main()