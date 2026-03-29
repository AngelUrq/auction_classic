#!/usr/bin/env python3

import os
import logging
import argparse

import sys
import requests
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from src.data.utils import create_access_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BLIZZARD_REGION = os.environ.get("BLIZZARD_REGION", "us")
BLIZZARD_LOCALE = os.environ.get("BLIZZARD_LOCALE", "en_US")
ITEM_API_URL = "https://{region}.api.blizzard.com/data/wow/item/{item_id}?namespace=static-{region}&locale={locale}&access_token={token}"


def connect():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


def fetch_missing_item_ids(cursor, limit):
    cursor.execute("""
        SELECT DISTINCT item_id FROM (
            SELECT item_id FROM auctions
            UNION
            SELECT item_id FROM commodities
        ) AS all_items
        WHERE item_id NOT IN (SELECT item_id FROM items)
        LIMIT %s
    """, (limit,))
    return [row[0] for row in cursor.fetchall()]


def fetch_item(item_id, token):
    url = ITEM_API_URL.format(
        region=BLIZZARD_REGION,
        item_id=item_id,
        locale=BLIZZARD_LOCALE,
        token=token,
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def parse_item(data):
    return (
        data["id"],
        data.get("name"),
        data.get("quality", {}).get("type"),
        data.get("level"),
        data.get("required_level"),
        data.get("item_class", {}).get("name"),
        data.get("item_subclass", {}).get("name"),
        data.get("inventory_type", {}).get("type"),
        data.get("purchase_price", 0) / 10000.0,
        data.get("sell_price", 0) / 10000.0,
        data.get("max_count"),
        data.get("purchase_quantity"),
        data.get("is_equippable"),
        data.get("is_stackable"),
        Json(data),
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch missing items from Blizzard API into PostgreSQL")
    parser.add_argument("--limit", type=int, default=100, help="Max number of items to fetch per run")
    args = parser.parse_args()

    conn = connect()
    token = create_access_token(
        os.environ["BLIZZARD_CLIENT_KEY"],
        os.environ["BLIZZARD_SECRET_KEY"],
        BLIZZARD_REGION,
    )["access_token"]

    try:
        with conn.cursor() as cursor:
            item_ids = fetch_missing_item_ids(cursor, args.limit)

        log.info(f"Fetching {len(item_ids)} missing items")

        rows = []
        for item_id in item_ids:
            try:
                data = fetch_item(item_id, token)
                rows.append(parse_item(data))
                log.info(f"Fetched item {item_id}: {data.get('name')}")
            except Exception as e:
                log.warning(f"Failed to fetch item {item_id}: {e}")

        if rows:
            with conn.cursor() as cursor:
                execute_values(cursor, """
                    INSERT INTO items (
                        item_id, name, quality, item_level, required_level,
                        item_class, item_subclass, inventory_type,
                        purchase_price, sell_price, max_count, purchase_quantity,
                        is_equippable, is_stackable, details
                    ) VALUES %s
                    ON CONFLICT (item_id) DO NOTHING
                """, rows)
            conn.commit()
            log.info(f"Inserted {len(rows)} items")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
