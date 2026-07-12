"""Poll the Blizzard auction API every 10 minutes for 1 hour to detect when data refreshes."""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CLIENT_KEY = os.environ["BLIZZARD_CLIENT_KEY"]
SECRET_KEY = os.environ["BLIZZARD_SECRET_KEY"]
REALM_ID = os.environ["BLIZZARD_REALM_ID"]
REGION = os.environ.get("BLIZZARD_REGION", "us")
NAMESPACE = os.environ.get("BLIZZARD_NAMESPACE", f"dynamic-{REGION}")
LOCALE = os.environ.get("BLIZZARD_LOCALE", "en_US")

POLL_INTERVAL_SECONDS = 10 * 60  # 10 minutes
TOTAL_DURATION_SECONDS = 60 * 60  # 1 hour
SCRATCH_DIR = ROOT / "generated" / "_poll_scratch"


def _fetch_token() -> str:
    response = requests.post(
        f"https://{REGION}.battle.net/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(CLIENT_KEY, SECRET_KEY),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _fetch_auctions(token: str) -> dict:
    url = (
        f"https://{REGION}.api.blizzard.com/data/wow/connected-realm"
        f"/{REALM_ID}/auctions"
        f"?namespace={NAMESPACE}&locale={LOCALE}"
    )
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _stable_hash(data: dict) -> str:
    """Hash auction IDs + buyout prices as a lightweight change fingerprint."""
    auctions = data.get("auctions", [])
    key = sorted((a.get("id"), a.get("buyout")) for a in auctions)
    blob = json.dumps(key, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _save_snapshot(data: dict, timestamp: datetime) -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_DIR / f"{timestamp.strftime('%Y%m%dT%H%M')}.json"
    path.write_text(json.dumps(data))
    return path


def main() -> None:
    print(f"Starting poll — every {POLL_INTERVAL_SECONDS // 60} min for {TOTAL_DURATION_SECONDS // 60} min")
    print(f"Realm: {REALM_ID}  Region: {REGION}\n")

    previous_hash: str | None = None
    previous_path: Path | None = None
    previous_auction_count: int | None = None
    start = time.monotonic()
    poll_number = 0

    while True:
        poll_number += 1
        now = datetime.now()
        print(f"[{now:%H:%M:%S}] Poll #{poll_number} — fetching token...")

        try:
            token = _fetch_token()
            print(f"[{now:%H:%M:%S}] Downloading auctions...")
            data = _fetch_auctions(token)
            raw_size = len(json.dumps(data).encode())
            if raw_size < 10 * 1024 * 1024:
                raise ValueError(f"Response too small ({raw_size / 1024 / 1024:.1f} MB < 10 MB) — likely bad data")
        except Exception as exc:
            print(f"[{now:%H:%M:%S}] ERROR: {exc}")
        else:
            current_hash = _stable_hash(data)
            current_count = len(data.get("auctions", []))
            snapshot_path = _save_snapshot(data, now)

            changed = previous_hash is not None and current_hash != previous_hash
            status = "CHANGED" if changed else ("(baseline)" if previous_hash is None else "no change")
            print(
                f"[{now:%H:%M:%S}] auctions={current_count:,}  hash={current_hash[:12]}  → {status}"
            )
            if changed:
                print(
                    f"             previous count={previous_auction_count:,}  "
                    f"delta={current_count - previous_auction_count:+,}"
                )

            # Delete the previous snapshot now that we have a newer one
            if previous_path is not None and previous_path.exists():
                previous_path.unlink()

            previous_hash = current_hash
            previous_path = snapshot_path
            previous_auction_count = current_count

        elapsed = time.monotonic() - start
        if elapsed >= TOTAL_DURATION_SECONDS:
            break

        sleep_for = POLL_INTERVAL_SECONDS - (time.monotonic() - start - (poll_number - 1) * POLL_INTERVAL_SECONDS)
        sleep_for = max(0, sleep_for)
        next_poll = datetime.fromtimestamp(time.time() + sleep_for)
        print(f"             sleeping {sleep_for:.0f}s — next poll at {next_poll:%H:%M:%S}\n")
        time.sleep(sleep_for)

    # Clean up the last snapshot
    if previous_path is not None and previous_path.exists():
        previous_path.unlink()
    if SCRATCH_DIR.exists() and not any(SCRATCH_DIR.iterdir()):
        SCRATCH_DIR.rmdir()

    print("\nDone.")


if __name__ == "__main__":
    main()
