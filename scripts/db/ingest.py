#!/usr/bin/env python3

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def connect():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


def is_processed(cursor, filename):
    cursor.execute("SELECT 1 FROM processed_files WHERE filename = %s", (filename,))
    return cursor.fetchone() is not None


def mark_processed(cursor, filename):
    cursor.execute(
        "INSERT INTO processed_files (filename) VALUES (%s) ON CONFLICT DO NOTHING",
        (filename,),
    )


def ingest_auctions_file(cursor, filepath, snapshot_ts):
    with open(filepath) as f:
        data = json.load(f)

    auctions = []
    observations = []
    bonuses = []
    modifiers = []

    for auction in data.get("auctions", []):
        item = auction["item"]

        if "pet_species_id" in item:
            continue

        auction_id = auction["id"]
        item_id = item["id"]
        context = item.get("context")
        buyout = auction.get("buyout", 0) / 10000.0 if auction.get("buyout") else None
        bid = auction.get("bid", 0) / 10000.0 if auction.get("bid") else None
        quantity = auction["quantity"]
        time_left = auction["time_left"]

        auctions.append((auction_id, item_id, context, buyout, bid, quantity, snapshot_ts, snapshot_ts))
        observations.append((auction_id, snapshot_ts, time_left))

        for bonus_id in item.get("bonus_lists", []):
            bonuses.append((auction_id, bonus_id))

        for modifier in item.get("modifiers", []):
            modifiers.append((auction_id, modifier["type"], modifier["value"]))

    if auctions:
        execute_values(cursor, """
            INSERT INTO auctions (auction_id, item_id, context, buyout, bid, quantity, first_seen_ts, last_seen_ts)
            VALUES %s
            ON CONFLICT (auction_id) DO UPDATE SET
                first_seen_ts = LEAST(auctions.first_seen_ts, EXCLUDED.first_seen_ts),
                last_seen_ts = GREATEST(auctions.last_seen_ts, EXCLUDED.last_seen_ts)
        """, auctions)

    if observations:
        execute_values(cursor, """
            INSERT INTO auction_observations (auction_id, snapshot_ts, time_left)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, observations)

    if bonuses:
        execute_values(cursor, """
            INSERT INTO auction_bonus (auction_id, bonus_id)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, bonuses)

    if modifiers:
        execute_values(cursor, """
            INSERT INTO auction_modifiers (auction_id, modifier_type, modifier_value)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, modifiers)

    log.info(f"Auctions: inserted {len(auctions)} auctions, {len(observations)} observations from {filepath.name}")


def ingest_commodities_file(cursor, filepath, snapshot_ts):
    with open(filepath) as f:
        data = json.load(f)

    commodities = []
    observations = []

    for auction in data.get("auctions", []):
        auction_id = auction["id"]
        item_id = auction["item"]["id"]
        unit_price = auction["unit_price"] / 10000.0
        quantity = auction["quantity"]
        time_left = auction["time_left"]

        commodities.append((auction_id, item_id, unit_price))
        observations.append((auction_id, snapshot_ts, quantity, time_left))

    if commodities:
        execute_values(cursor, """
            INSERT INTO commodities (auction_id, item_id, unit_price)
            VALUES %s
            ON CONFLICT (auction_id) DO NOTHING
        """, commodities)

    if observations:
        execute_values(cursor, """
            INSERT INTO commodity_observations (auction_id, snapshot_ts, quantity, time_left)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, observations)

    log.info(f"Commodities: inserted {len(commodities)} commodities, {len(observations)} observations from {filepath.name}")


def find_json_files(data_dir):
    return sorted(Path(data_dir).rglob("*.json"))


def parse_snapshot_ts(filepath):
    stem = filepath.stem  # e.g. 20260328T14
    return datetime.strptime(stem, "%Y%m%dT%H")


def ingest_directory(conn, data_dir, ingest_fn):
    files = find_json_files(data_dir)
    log.info(f"Found {len(files)} files in {data_dir}")

    for filepath in files:
        filename = str(filepath)

        with conn.cursor() as cursor:
            if is_processed(cursor, filename):
                log.debug(f"Skipping already processed file: {filepath.name}")
                continue

        try:
            snapshot_ts = parse_snapshot_ts(filepath)
        except ValueError:
            log.warning(f"Skipping file with unexpected name format: {filepath.name}")
            continue

        try:
            with conn.cursor() as cursor:
                ingest_fn(cursor, filepath, snapshot_ts)
                mark_processed(cursor, filename)
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error(f"Failed to process {filepath.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ingest auction JSON files into PostgreSQL")
    parser.add_argument("--auctions-dir", default=os.environ.get("SERVER_AUCTIONS_DIR"), help="Path to auctions JSON directory")
    parser.add_argument("--commodities-dir", default=os.environ.get("SERVER_COMMODITIES_DIR"), help="Path to commodities JSON directory")
    args = parser.parse_args()

    conn = connect()

    try:
        if args.auctions_dir:
            ingest_directory(conn, args.auctions_dir, ingest_auctions_file)
        else:
            log.warning("No auctions directory specified, skipping")

        if args.commodities_dir:
            ingest_directory(conn, args.commodities_dir, ingest_commodities_file)
        else:
            log.warning("No commodities directory specified, skipping")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
