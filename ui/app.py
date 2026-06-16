import torch
import sys
import os
import threading
import pandas as pd
import json
import numpy as np
import gradio as gr

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from tqdm import tqdm
from src.models.inference import predict_dataframe
from src.data.utils import load_auctions_from_sample
from src.models.auction_transformer import AuctionTransformer

model_loaded = False
df_auctions = None
model = None
feature_stats = None
prediction_time = None
recommendations = None
ckpt_path = './models/transformer-4.2M-survival_24-lr3e-05-bs128/last.ckpt'
max_hours_back = 24
max_sequence_length = 1024
sold_threshold = 8

item_to_idx = None
context_to_idx = None
bonus_to_idx = None
modtype_to_idx = None
idx_to_item = None

AH_CUT = 0.05        # auction house cut taken from the sale price
UNDERCUT = 0.0005    # relist just under the next-cheapest competitor to become the cheapest (rank 0)

TIME_LEFT_MAPPING = {
    'VERY_LONG': 48,
    'LONG': 12,
    'MEDIUM': 2,
    'SHORT': 0.5
}


def _add_wowhead_links(df):
    if df.empty or 'item_id' not in df.columns:
        return df
    df = df.copy()
    df['item_id'] = df['item_id'].apply(
        lambda x: f'<a href="https://www.wowhead.com/item={x}/" target="_blank">{x}</a>' if pd.notna(x) else x
    )
    return df


def _reload_data():
    """Reload auction data without reloading the model."""
    global df_auctions, prediction_time

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data/auctions/")

    files = []
    for root, _, fs in os.walk(data_path):
        for fn in fs:
            if fn.endswith('.json'):
                try:
                    dt = datetime.strptime(fn.split('.')[0], '%Y%m%dT%H')
                    files.append(dt)
                except Exception:
                    continue

    new_prediction_time = max(files) if files else datetime.now()
    if new_prediction_time == prediction_time:
        print(f"Data already up to date ({prediction_time}), skipping reload.")
        return

    print(f"Reloading auction data for {new_prediction_time}...")
    new_df = load_auctions_from_sample(
        data_path,
        new_prediction_time,
        TIME_LEFT_MAPPING,
        item_to_idx,
        context_to_idx,
        bonus_to_idx,
        modtype_to_idx,
        max_hours_back=max_hours_back,
        include_targets=False
    )
    new_df['item_id'] = new_df['item_index'].map(idx_to_item)
    print(f"Reloaded {len(new_df)} auctions as of {new_prediction_time}")

    # Atomic reference swap: any in-flight computation that already captured
    # the old df/prediction_time into locals is unaffected.
    df_auctions = new_df
    prediction_time = new_prediction_time


def _start_background_reload(reload_minute=3):
    def seconds_until_next_reload():
        now = datetime.now()
        next_reload = now.replace(second=0, microsecond=0, minute=reload_minute)
        if next_reload <= now:
            next_reload += timedelta(hours=1)
        return (next_reload - now).total_seconds()

    def loop():
        while True:
            delay = seconds_until_next_reload()
            print(f"Next data reload in {delay:.0f}s (at :{reload_minute:02d})")
            threading.Event().wait(delay)
            try:
                _reload_data()
            except Exception as e:
                print(f"Background data reload failed: {e}")

    t = threading.Thread(target=loop, daemon=True)
    t.start()


def load_data_and_model():
    global model_loaded, model, feature_stats
    global item_to_idx, context_to_idx, bonus_to_idx, modtype_to_idx, idx_to_item

    if model_loaded:
        return

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mappings_dir = os.path.join(base_path, 'generated/mappings')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join(mappings_dir, 'item_to_idx.json'), 'r') as f:
        item_to_idx = json.load(f)
    with open(os.path.join(mappings_dir, 'context_to_idx.json'), 'r') as f:
        context_to_idx = json.load(f)
    with open(os.path.join(mappings_dir, 'bonus_to_idx.json'), 'r') as f:
        bonus_to_idx = json.load(f)
    with open(os.path.join(mappings_dir, 'modtype_to_idx.json'), 'r') as f:
        modtype_to_idx = json.load(f)
    idx_to_item = {v: k for k, v in item_to_idx.items()}

    feature_stats = torch.load(os.path.join(base_path, 'generated/feature_stats.pt'))

    model = AuctionTransformer.load_from_checkpoint(
        os.path.join(base_path, ckpt_path),
        map_location=device,
        weights_only=False
    )
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')
    model.eval()
    print('Pre-trained Transformer model loaded successfully.')

    _reload_data()
    model_loaded = True
    return "Model and data loaded successfully!"



def generate_recommendations(min_profit, min_sale_probability, hold_horizon_hours):
    """Recommend flips with the become-cheapest + expected-value strategy.

    For each item we buy the cheapest listing and relist just under the next-cheapest
    competitor, so our relisting becomes the cheapest (buyout_rank 0). The model scores
    that exact counterfactual listing and predicts the probability it sells within the
    hold horizon. We keep flips whose post-fee margin and sale probability clear the
    thresholds, then rank by expected value (margin x sale_probability), breaking ties
    toward faster expected sales (capital velocity).
    """
    global model_loaded

    if not model_loaded:
        load_data_and_model()

    # Snapshot globals so the entire computation sees a consistent dataset even
    # if the background thread swaps df_auctions/prediction_time mid-run.
    df = df_auctions
    pred_time = prediction_time
    mdl = model
    stats = feature_stats

    df_now = df[df["snapshot_offset"] == 0].copy()
    if df_now.empty:
        return pd.DataFrame()

    recommendations_list = []

    for item_index, df_now_item in tqdm(df_now.groupby("item_index")):
        # Need a listing to buy (cheapest) and a competitor to undercut (next-cheapest).
        sorted_buyouts = df_now_item["buyout"].sort_values()
        if len(sorted_buyouts) < 2:
            continue

        cheapest_index = sorted_buyouts.index[0]
        cheapest_row = df_now_item.loc[cheapest_index].copy()
        cheapest_buyout = float(sorted_buyouts.iloc[0])
        next_cheapest = float(sorted_buyouts.iloc[1])
        auction_id = cheapest_row["id"]

        # Become the new cheapest after buying #1; gate on the post-fee margin first.
        relist_price = next_cheapest * (1 - UNDERCUT)
        margin = relist_price * (1 - AH_CUT) - cheapest_buyout
        if margin < min_profit:
            continue

        # Full per-item context (0..max_hours_back); drop the bought auction at the
        # current step and replace it with our fresh relisting at relist_price.
        df_item_full = df[
            (df["item_index"] == item_index) &
            (df["snapshot_offset"] >= 0) &
            (df["snapshot_offset"] <= max_hours_back)
        ].copy()
        df_item_full = df_item_full[~((df_item_full["id"] == auction_id) & (df_item_full["snapshot_offset"] == 0))]

        relist_row = cheapest_row.copy()
        relist_row["id"] = auction_id
        relist_row["buyout"] = float(relist_price)
        relist_row["bid"] = 0.0
        relist_row["time_left"] = 48.0
        relist_row["listing_age"] = 0.0
        relist_row["snapshot_time"] = pred_time
        relist_row["snapshot_offset"] = 0
        relist_row["hour_of_week"] = pred_time.weekday() * 24 + pred_time.hour

        df_item_full = pd.concat([df_item_full, pd.DataFrame([relist_row])], ignore_index=True)

        prediction_df = predict_dataframe(
            model=mdl,
            df_auctions=df_item_full,
            prediction_time=pred_time,
            feature_stats=stats,
            max_hours_back=max_hours_back,
            max_sequence_length=max_sequence_length,
            quick_sale_threshold_hours=hold_horizon_hours,
        )
        if prediction_df.empty:
            continue

        # Prediction for our relisted auction: same id also appears in the historical
        # snapshots (offset > 0), which predict_dataframe leaves unscored (NaN), so we
        # must select the offset-0 relist row specifically.
        pred_row = prediction_df[
            (prediction_df["id"] == auction_id) & (prediction_df["snapshot_offset"] == 0)
        ].iloc[0]
        sale_probability = float(pred_row["sale_probability"])
        if sale_probability < min_sale_probability:
            continue

        expected_value = margin * sale_probability

        recommendation = pd.Series({
            "item_id": cheapest_row.get("item_id"),
            "buyout": round(cheapest_buyout, 2),
            "next_cheapest": round(next_cheapest, 2),
            "relist_price": round(relist_price, 2),
            "margin": round(margin, 2),
            "sale_probability": round(sale_probability, 3),
            "expected_value": round(expected_value, 2),
            "expected_duration": float(pred_row["expected_duration"]),
            "prediction_q10": float(pred_row["prediction_q10"]),
            "prediction_q50": float(pred_row["prediction_q50"]),
            "prediction_q90": float(pred_row["prediction_q90"]),
            "quantity": pred_row.get("quantity", cheapest_row.get("quantity")),
        })
        recommendations_list.append(recommendation)

    if not recommendations_list:
        return pd.DataFrame()

    return pd.DataFrame(recommendations_list).sort_values(
        ["expected_value", "expected_duration"],
        ascending=[False, True],
    )

RECOMMENDATIONS_LOG_DIR = Path(__file__).resolve().parent.parent / "generated" / "logs" / "recommendations"


def _log_recommendations(df: pd.DataFrame, min_profit: float, min_sale_probability: float, hold_horizon: float) -> None:
    logged = df.copy()
    logged["logged_at"] = datetime.now().isoformat()
    logged["min_profit"] = min_profit
    logged["min_sale_probability"] = min_sale_probability
    logged["hold_horizon"] = hold_horizon
    RECOMMENDATIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RECOMMENDATIONS_LOG_DIR / f"{datetime.now().date()}.csv"
    logged.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def generate_recommendations_ui(min_profit, min_sale_probability, hold_horizon):
    """UI function for generating recommendations"""
    global recommendations

    recommendations = generate_recommendations(
        float(min_profit), float(min_sale_probability), float(hold_horizon)
    )

    if recommendations.empty:
        return "No recommendations found. Try lowering the minimum margin or sale probability.", pd.DataFrame()

    _log_recommendations(recommendations, float(min_profit), float(min_sale_probability), float(hold_horizon))
    return f"Found {len(recommendations)} recommendations!", _add_wowhead_links(recommendations)

def create_ui():
    """Create the Gradio interface"""
    with gr.Blocks(title="Auction House Data", theme=gr.themes.Default()) as app:
        gr.Markdown("# Auction House Data")
        with gr.Tab("Current Auctions"):
            gr.Markdown("## Current Auctions")

            with gr.Row():
                item_search = gr.Textbox(label="Search by Item ID", placeholder="Enter item ID...")
                search_button = gr.Button("Search")
                time_offset_filter = gr.Slider(label="Display Time Offset (hours ago)", minimum=0, maximum=48, value=0, step=1)
                predict_button = gr.Button("Predict Sale Probability")
                store_csv_button = gr.Button("Store CSV")

            display_columns = ['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_ids', 'modifier_types', 'modifier_values', 'listing_age', 'snapshot_offset']

            initial_display = pd.DataFrame()
            if df_auctions is not None:
                current_slice = df_auctions[df_auctions['snapshot_offset'] == 0]
                initial_display = current_slice[display_columns].head(100)

            auctions_display = gr.Dataframe(_add_wowhead_links(initial_display), interactive=False, datatype=["html"] + ["str"] * 14)
            store_csv_status = gr.Markdown("")

            def filter_auctions(item_id, time_offset_value):
                df = df_auctions
                if df is None:
                    return pd.DataFrame()

                item_id = (item_id or "").strip()
                time_offset_value = float(time_offset_value or 0)
                filtered_df = df[df['snapshot_offset'] == time_offset_value]
                if item_id:
                    filtered_df = filtered_df[filtered_df['item_id'] == item_id]

                return _add_wowhead_links(filtered_df[display_columns].head(100))

            def predict_filtered_auctions(item_id, time_offset_value):
                df = df_auctions
                pred_time = prediction_time
                mdl = model
                stats = feature_stats

                if df is None:
                    return pd.DataFrame()

                item_id = (item_id or "").strip()
                time_offset_value = float(time_offset_value or 0)

                if not item_id:
                    filtered_df = df.head(100).copy()
                else:
                    filtered_df = df[df['item_id'] == item_id].copy()

                if filtered_df.empty:
                    return pd.DataFrame()

                prediction_results = predict_dataframe(
                    mdl,
                    filtered_df,
                    pred_time,
                    stats,
                    max_hours_back=max_hours_back,
                    max_sequence_length=max_sequence_length
                )

                prediction_results = prediction_results[prediction_results['snapshot_offset'] == 0]

                if prediction_results.empty:
                    return pd.DataFrame(columns=['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_ids', 'modifier_types', 'modifier_values', 'listing_age', 'prediction_q10', 'prediction_q50', 'prediction_q90', 'expected_duration', 'sale_probability'])

                display_columns_prediction = display_columns + ['prediction_q10', 'prediction_q50', 'prediction_q90', 'expected_duration', 'sale_probability']
                return _add_wowhead_links(prediction_results[display_columns_prediction].head(100))

            def store_item_csv(item_id, _time_offset_value):
                df = df_auctions
                if df is None:
                    return "No auctions loaded yet."

                item_id = (item_id or "").strip()
                if not item_id:
                    return "Please provide an item ID before storing."

                item_df = df[df['item_id'] == item_id].copy()
                if item_df.empty:
                    return f"No rows found for item {item_id}."

                item_df = item_df.sort_values("listing_age")

                output_dir = Path(__file__).resolve().parent.parent / "generated"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / f"{item_id}.csv"
                item_df.to_csv(output_path, index=False)

                return f"Stored {len(item_df)} rows to {output_path}."

            search_button.click(
                filter_auctions,
                inputs=[item_search, time_offset_filter],
                outputs=[auctions_display]
            )

            time_offset_filter.change(
                filter_auctions,
                inputs=[item_search, time_offset_filter],
                outputs=[auctions_display]
            )

            predict_button.click(
                predict_filtered_auctions,
                inputs=[item_search, time_offset_filter],
                outputs=[auctions_display]
            )

            store_csv_button.click(
                store_item_csv,
                inputs=[item_search, time_offset_filter],
                outputs=[store_csv_status]
            )

        with gr.Tab("Recommendations"):
            gr.Markdown("## Auction Recommendations")
            gr.Markdown(
                "**Become-the-cheapest** strategy: buy the cheapest listing, then relist just under the "
                "next-cheapest competitor so your auction is the cheapest. The model scores that exact "
                "relisting and predicts how likely it is to **sell within your hold horizon**. Flips are "
                "ranked by **expected value** (post-fee margin × sale probability)."
            )
            gr.Markdown("**Columns:**")
            gr.Markdown("- **buyout**: cost to buy the current cheapest listing")
            gr.Markdown("- **next_cheapest / relist_price**: the competitor you undercut, and your resulting price")
            gr.Markdown(f"- **margin**: profit after the {int(AH_CUT * 100)}% AH cut if it sells (relist_price × {1 - AH_CUT:g} − buyout)")
            gr.Markdown("- **sale_probability**: model P(sells within the hold horizon)")
            gr.Markdown("- **expected_value**: margin × sale_probability (ranking key)")
            gr.Markdown("- **expected_duration / prediction_q10·q50·q90**: predicted hours to sell")

            with gr.Row():
                min_profit_input = gr.Number(label="Minimum Margin After Fees (gold)", value=100, minimum=0, step=100)
                min_sale_probability_input = gr.Slider(label="Minimum Sale Probability", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
                hold_horizon_input = gr.Slider(label="Hold Horizon (hours)", minimum=1, maximum=24, value=12, step=1)

            generate_button = gr.Button("Generate Recommendations")
            result_text = gr.Markdown()
            recommendations_output = gr.Dataframe(datatype=["html"] + ["str"] * 11)

            generate_button.click(
                generate_recommendations_ui,
                inputs=[min_profit_input, min_sale_probability_input, hold_horizon_input],
                outputs=[result_text, recommendations_output]
            )

        with gr.Tab("Individual Flipping"):
            gr.Markdown("## Individual Item Flipping Recommendations")
            gr.Markdown("**Prediction columns explained:**")
            gr.Markdown("- **predicted_hours_q10**: 10th percentile (optimistic timing)")
            gr.Markdown("- **predicted_hours_q50**: 50th percentile (median hours to sale)")
            gr.Markdown("- **predicted_hours_q90**: 90th percentile (pessimistic timing)")
            gr.Markdown("- **is_sold**: Probability (0-1) that the item will sell")

            with gr.Row():
                flip_item_search = gr.Textbox(label="Search by Item ID", placeholder="Enter item ID...")
                flip_search_button = gr.Button("Search")

            item_auctions_display = gr.Dataframe(interactive=False, datatype=["html"] + ["str"] * 7)

            with gr.Row():
                custom_buyout = gr.Number(label="Custom Buyout Price (gold)", value=0, minimum=0, step=100)
                predict_flip_button = gr.Button("Predict Flip")

            flip_result = gr.Dataframe(interactive=False, datatype=["html"] + ["str"] * 8)

            def search_item_for_flip(item_id):
                global model_loaded

                if not model_loaded:
                    load_data_and_model()

                df = df_auctions
                item_id = (item_id or "").strip()
                if not item_id or df is None:
                    return pd.DataFrame()

                filtered_df = df[
                    (df['item_id'] == item_id) &
                    (df['snapshot_offset'] == 0)
                ]
                flip_display_columns = ['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_ids', 'modifier_types', 'modifier_values']
                return _add_wowhead_links(filtered_df[flip_display_columns])

            def predict_item_flip(item_id, custom_price):
                global model_loaded

                if not model_loaded:
                    load_data_and_model()

                df = df_auctions
                pred_time = prediction_time
                mdl = model
                stats = feature_stats

                item_id = (item_id or "").strip()
                if not item_id or df is None:
                    return pd.DataFrame()

                item_df = df[df['item_id'] == item_id].copy()

                if item_df.empty:
                    print("No auctions found for this item")
                    return pd.DataFrame()

                min_buyout_idx = item_df['buyout'].astype(float).idxmin()
                lowest_auction = item_df.loc[min_buyout_idx].copy()

                target_price = float(custom_price or 0)
                if target_price <= 0:
                    target_price = float(lowest_auction['buyout'])

                auction_id = lowest_auction['id']
                item_index = lowest_auction['item_index']

                df_item_full = df[
                    (df['item_index'] == item_index) &
                    (df['snapshot_offset'] >= 0) &
                    (df['snapshot_offset'] <= max_hours_back)
                ].copy()

                df_item_full = df_item_full[~((df_item_full['id'] == auction_id) & (df_item_full['snapshot_offset'] == 0))]

                # New relist row mirrors recommendation flow
                relist_row = lowest_auction.copy()
                relist_row['buyout'] = float(target_price)
                relist_row['bid'] = 0.0
                relist_row['time_left'] = 48.0
                relist_row['listing_age'] = 0.0
                relist_row['snapshot_time'] = pred_time
                relist_row['snapshot_offset'] = 0
                relist_row['hour_of_week'] = pred_time.weekday() * 24 + pred_time.hour

                df_item_full = pd.concat([df_item_full, pd.DataFrame([relist_row])], ignore_index=True)

                prediction_df = predict_dataframe(
                    mdl,
                    df_item_full,
                    pred_time,
                    stats,
                    max_hours_back=max_hours_back,
                    max_sequence_length=max_sequence_length
                )

                if prediction_df.empty:
                    print("No prediction found for this item")
                    return pd.DataFrame()

                flip_prediction = prediction_df[
                    (prediction_df['id'] == auction_id) &
                    (~prediction_df['prediction_q50'].isna())
                ]

                if flip_prediction.empty:
                    print("No prediction found for this item")
                    return pd.DataFrame()

                flip_prediction = flip_prediction.iloc[0]

                result = pd.DataFrame([{
                    'item_id': lowest_auction['item_id'],
                    'quantity': lowest_auction['quantity'],
                    'original_price': round(float(lowest_auction['buyout']), 2),
                    'custom_price': round(float(target_price), 2),
                    'potential_profit': round(float(target_price) - float(lowest_auction['buyout']), 2),
                    'predicted_hours_q10': flip_prediction['prediction_q10'],
                    'predicted_hours_q50': flip_prediction['prediction_q50'],
                    'predicted_hours_q90': flip_prediction['prediction_q90'],
                    'sale_probability': flip_prediction['sale_probability']
                }])

                return _add_wowhead_links(result)

            flip_search_button.click(
                search_item_for_flip,
                inputs=[flip_item_search],
                outputs=[item_auctions_display]
            )

            predict_flip_button.click(
                predict_item_flip,
                inputs=[flip_item_search, custom_buyout],
                outputs=[flip_result]
            )


    return app

if __name__ == "__main__":
    load_data_and_model()
    _start_background_reload(reload_minute=3)

    app = create_ui()
    app.launch(share=True)
