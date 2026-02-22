#!/usr/bin/env python3
"""
Check for duplicate auction JSON files by comparing MD5 hashes.
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1 MB


def md5(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def collect_files(base: Path) -> list[Path]:
    today = date.today().strftime("%Y%m%d")
    files = []
    for f in sorted(base.rglob("*.json")):
        if f.stem[:8] != today:
            files.append(f)
    return files


def main():
    parser = argparse.ArgumentParser(description="Check for duplicate auction JSON files.")
    parser.add_argument("--data_dir", type=str, default="data/auctions/", help="Path to the auctions folder")
    args = parser.parse_args()

    base = Path(args.data_dir)
    if not base.exists():
        print(f"ERROR: directory not found: {base}")
        sys.exit(1)

    files = collect_files(base)
    if not files:
        print("ERROR: no files found.")
        sys.exit(1)

    hashes = defaultdict(list)
    for f in tqdm(files, desc="Hashing files"):
        hashes[md5(f)].append(f)

    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}

    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate group(s):\n")
        for h, paths in duplicates.items():
            print(f"  {h}")
            for p in paths:
                print(f"    {p}")
    else:
        print("\nNo duplicates found.")

    print(f"\nTotal files checked : {len(files)}")
    print(f"Unique hashes       : {len(hashes)}")
    print(f"Duplicate groups    : {len(duplicates)}")
    print(f"Duplicated files    : {sum(len(p) for p in duplicates.values())}")

    sys.exit(1 if duplicates else 0)


if __name__ == "__main__":
    main()
