#!/usr/bin/env python3
"""
Validate an auctions directory integrity.

Checks per day:
  - Exactly 24 hourly JSON files present
  - Each JSON file is parseable
  - Each JSON file is at least 5 MB
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from tqdm import tqdm

MIN_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
EXPECTED_HOURS = 24


def validate_day(day_dir: Path) -> dict:
    year, month, day = day_dir.parts[-3], day_dir.parts[-2], day_dir.parts[-1]
    date_prefix = f"{year}{month}{day}"

    results = {
        "date": f"{year}/{month}/{day}",
        "file_count": 0,
        "missing_hours": [],
        "invalid_json": [],
        "too_small": [],
        "ok_files": 0,
        "total_size_bytes": 0,
    }

    for hour in range(24):
        filename = f"{date_prefix}T{hour:02d}.json"
        filepath = day_dir / filename

        if not filepath.exists():
            results["missing_hours"].append(hour)
            continue

        results["file_count"] += 1
        size = filepath.stat().st_size
        results["total_size_bytes"] += size

        if size < MIN_FILE_SIZE_BYTES:
            results["too_small"].append((hour, size))
            continue

        try:
            with open(filepath) as f:
                json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            results["invalid_json"].append((hour, str(e)))
            continue

        results["ok_files"] += 1

    return results


def collect_day_dirs(base: Path) -> list[Path]:
    days = []
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if day_dir.is_dir():
                    days.append(day_dir)
    return days


def format_size(n_bytes: int) -> str:
    return f"{n_bytes / 1024 / 1024:.1f} MB"


def main():
    parser = argparse.ArgumentParser(description="Validate auctions data directory integrity.")
    parser.add_argument("--data_dir", type=str, default="data/auctions/", help="Path to the auctions folder")
    args = parser.parse_args()

    base = Path(args.data_dir)
    if not base.exists():
        print(f"ERROR: directory not found: {base}")
        sys.exit(1)

    today = date.today().strftime("%Y/%m/%d")
    day_dirs = [d for d in collect_day_dirs(base) if f"{d.parts[-3]}/{d.parts[-2]}/{d.parts[-1]}" != today]
    if not day_dirs:
        print("ERROR: no day directories found.")
        sys.exit(1)

    all_results = []
    for day_dir in tqdm(day_dirs, desc="Checking days"):
        r = validate_day(day_dir)
        all_results.append(r)
        if r["missing_hours"] or r["invalid_json"] or r["too_small"]:
            tqdm.write(f"  {r['date']}")
            if r["missing_hours"]:
                hrs = ", ".join(f"{h:02d}:00" for h in r["missing_hours"])
                tqdm.write(f"    Missing hours ({len(r['missing_hours'])}): {hrs}")
            for hour, size in r["too_small"]:
                tqdm.write(f"    Too small — hour {hour:02d}:00: {format_size(size)} (< 5 MB)")
            for hour, err in r["invalid_json"]:
                tqdm.write(f"    Invalid JSON — hour {hour:02d}:00: {err}")

    days_with_issues = [r for r in all_results if (
        r["missing_hours"] or r["invalid_json"] or r["too_small"]
    )]

    if not days_with_issues:
        print("All days passed validation.")
    else:
        print(f"Days with issues: {', '.join(r['date'] for r in days_with_issues)}")
    print()

    total_days = len(all_results)
    perfect_days = sum(
        1 for r in all_results
        if not r["missing_hours"] and not r["invalid_json"] and not r["too_small"]
    )
    days_missing_files = sum(1 for r in all_results if r["missing_hours"])
    total_missing_hours = sum(len(r["missing_hours"]) for r in all_results)
    total_small_files = sum(len(r["too_small"]) for r in all_results)
    total_invalid_json = sum(len(r["invalid_json"]) for r in all_results)
    total_ok_files = sum(r["ok_files"] for r in all_results)
    total_size = sum(r["total_size_bytes"] for r in all_results)

    file_sizes = [f.stat().st_size for d in day_dirs for f in d.glob("*.json")]
    avg_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
    min_size = min(file_sizes) if file_sizes else 0
    max_size = max(file_sizes) if file_sizes else 0

    print("Summary:")
    print(f"  Date range        : {all_results[0]['date']} → {all_results[-1]['date']}")
    print(f"  Total days        : {total_days}")
    print(f"  Perfect days      : {perfect_days} / {total_days}")
    print(f"  Days with missing : {days_missing_files}  ({total_missing_hours} missing hour-files)")
    print(f"  Files too small   : {total_small_files}")
    print(f"  Invalid JSON      : {total_invalid_json}")
    print(f"  Valid files       : {total_ok_files} / {total_days * EXPECTED_HOURS}")
    print()
    print(f"  Total data size   : {format_size(total_size)}")
    print(f"  Avg file size     : {format_size(avg_size)}")
    print(f"  Min file size     : {format_size(min_size)}")
    print(f"  Max file size     : {format_size(max_size)}")

    sys.exit(1 if days_with_issues else 0)


if __name__ == "__main__":
    main()
