"""Generate flip recommendations for cron use, printing them to stdout.

Reuses the exact become-the-cheapest recommendation logic that powers the
Gradio Recommendations tab (``ui/app.py``), so the two never drift. Intended to
be run hourly (just after the data refresh) from cron; for now it only prints
the ranked flips, but a notification hook can be added at the end later.

sale_probability is the model's P(genuine sale, is_sold) over the hold horizon (default 48h =
"will it sell at all"), post-hoc calibrated against the is_sold proxy when
generated/sale_calibrator.pkl exists (raw/optimistic otherwise). Read expected_value as a ranking.

Usage:
    python scripts/generate_recommendations.py \
        --min-profit 100 --min-sale-probability 0.6 --hold-horizon 48 --top 25
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
# ui/app.py isn't a package; add its directory so we can import it directly. Its
# own module-level sys.path.append handles the `src` imports.
sys.path.append(str(ROOT / "ui"))

import app as ui_app  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-profit",
        type=float,
        default=10.0,
        help="Minimum post-fee margin in gold to keep a flip (default: 100).",
    )
    parser.add_argument(
        "--min-sale-probability",
        type=float,
        default=0.6,
        help="Minimum P(genuine sale, is_sold, within the hold horizon) to keep a flip. "
             "Calibrated against the is_sold proxy when a calibrator is present (default: 0.6).",
    )
    parser.add_argument(
        "--hold-horizon",
        type=float,
        default=48.0,
        help="Hold horizon in hours for the sale-probability estimate; 48 = full window = "
             "P(sold at all) (default: 48).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Print only the top N recommendations by expected value (default: 25; 0 = all).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip appending the run to generated/logs/recommendations/.",
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="Skip posting the recommendations to Slack (SLACK_WEBHOOK).",
    )
    return parser.parse_args()


def _print_recommendations(recommendations: pd.DataFrame, top: int) -> None:
    display = recommendations if top <= 0 else recommendations.head(top)
    with pd.option_context(
        "display.max_rows", None,
        "display.width", None,
        "display.max_columns", None,
    ):
        print(display.to_string(index=False))


def _format_slack_message(recommendations: pd.DataFrame, top: int, args: argparse.Namespace) -> str:
    """Build a Slack mrkdwn message: a header plus one linked line per top flip.

    Slack has no table block, and links don't render inside a monospace code
    fence, so each flip is one bullet with a clickable wowhead item link.
    """
    display = recommendations if top <= 0 else recommendations.head(top)
    header = (
        f"*{len(recommendations)} flip recommendation(s)* "
        f"(min_profit={args.min_profit:g}, min_sale_prob={args.min_sale_probability:g}, "
        f"hold={args.hold_horizon:g}h) — data as of {ui_app.prediction_time}"
    )
    lines = [header]
    for _, row in display.iterrows():
        item_id = row.get("item_id")
        item_link = f"<https://www.wowhead.com/item={item_id}|{item_id}>" if pd.notna(item_id) else str(item_id)
        lines.append(
            f"• {item_link} — buy {row['buyout']:,.0f} → relist {row['relist_price']:,.0f} | "
            f"margin {row['margin']:,.0f} | P(sale) {row['sale_probability']:.2f} | "
            f"EV {row['expected_value']:,.0f} | ~{row['expected_duration']:.0f}h "
            f"(q10/50/90 {row['prediction_q10']:.0f}/{row['prediction_q50']:.0f}/{row['prediction_q90']:.0f}h)"
        )
    if 0 < top < len(recommendations):
        lines.append(f"_…and {len(recommendations) - top} more._")
    return "\n".join(lines)


def _notify_slack(text: str) -> None:
    """POST a message to the Slack incoming webhook in SLACK_WEBHOOK."""
    webhook = os.environ.get("SLACK_WEBHOOK")
    if not webhook:
        print("SLACK_WEBHOOK not set; skipping Slack notification.")
        return

    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
        print("Posted recommendations to Slack.")
    except Exception as error:
        print(f"Slack notification failed: {error}")


def main() -> None:
    args = _parse_args()

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Loading model and data...")
    ui_app.load_data_and_model()
    print(f"Data as of {ui_app.prediction_time}; generating recommendations...")

    recommendations = ui_app.generate_recommendations(
        args.min_profit, args.min_sale_probability, args.hold_horizon
    )

    if recommendations.empty:
        print("No recommendations found. Try lowering --min-profit or --min-sale-probability.")
        if not args.no_slack:
            _notify_slack(
                f"*0 flip recommendations this hour* "
                f"(min_profit={args.min_profit:g}, min_sale_prob={args.min_sale_probability:g}, "
                f"hold={args.hold_horizon:g}h) — data as of {ui_app.prediction_time}"
            )
        return

    print(f"\nFound {len(recommendations)} recommendations "
          f"(min_profit={args.min_profit:g}, min_sale_probability={args.min_sale_probability:g}, "
          f"hold_horizon={args.hold_horizon:g}h):\n")
    _print_recommendations(recommendations, args.top)

    if not args.no_log:
        ui_app._log_recommendations(
            recommendations, args.min_profit, args.min_sale_probability, args.hold_horizon
        )
        print(f"\nLogged run to {ui_app.RECOMMENDATIONS_LOG_DIR}")

    if not args.no_slack:
        _notify_slack(_format_slack_message(recommendations, args.top, args))


if __name__ == "__main__":
    main()
