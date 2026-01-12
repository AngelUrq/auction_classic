import torch
import sys
import os
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
ckpt_path = 'models/transformer-4.2M-quantile-historical_72-lr1e-04-bs64/last-v3.ckpt'
max_hours_back = 72
max_sequence_length = 4096
sold_threshold = 8


def load_data_and_model():
    """Load data and model only once"""
    global model_loaded, df_auctions, model, feature_stats, prediction_time
    
    if model_loaded:
        return
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data/tww/auctions/")

    # Run sync_auctions script on startup
    script_path = os.path.join(base_path, "scripts", "data_collection", "sync_auctions.sh")
    if os.path.exists(script_path):
        print(f"Running sync_auctions script at {script_path}")
        os.system(f"bash {script_path}")
    else:
        print(f"Error: Script not found at {script_path}")

    time_left_mapping = {
        'VERY_LONG': 48,
        'LONG': 12,
        'MEDIUM': 2,
        'SHORT': 0.5
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prediction_time = datetime.now()
    print(f'Prediction time: {prediction_time}')

    mappings_dir = os.path.join(base_path, 'generated/mappings')

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
        map_location=device
    )

    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')
    model.eval()
    print('Pre-trained Transformer model loaded successfully.')

    df_auctions = load_auctions_from_sample(
        data_path, 
        prediction_time, 
        time_left_mapping, 
        item_to_idx, 
        context_to_idx, 
        bonus_to_idx, 
        modtype_to_idx, 
        max_hours_back=max_hours_back,
        include_targets=False
    )
    df_auctions['item_id'] = df_auctions['item_index'].map(idx_to_item)
    print(f'Number of auctions loaded: {len(df_auctions)}')
    
    model_loaded = True
    return "Model and data loaded successfully!"


def generate_recommendations(expected_profit, median_discount=0.75):
    global df_auctions, model, feature_stats, prediction_time, model_loaded, max_hours_back

    if not model_loaded:
        load_data_and_model()

    df_now = df_auctions[df_auctions["time_offset"] == 0].copy()
    if df_now.empty:
        return pd.DataFrame()

    recommendations_list = []

    for item_index, df_now_item in tqdm(df_now.groupby("item_index")):
        # Cheapest present listing -> buy it
        cheapest_index = df_now_item["buyout"].idxmin()
        cheapest_row = df_auctions.loc[cheapest_index].copy()
        cheapest_buyout = float(cheapest_row["buyout"])
        auction_id = cheapest_row["id"]

        # Present-only q25 target price
        present_q25 = float(df_now_item["buyout"].quantile(0.25))
        target_price = present_q25 * 0.7
        potential_profit = target_price - cheapest_buyout
        if potential_profit < expected_profit:
            continue

        # Full per-item context (0..max_hours_back), drop this auction id EVERYWHERE
        df_item_full = df_auctions[
            (df_auctions["item_index"] == item_index) &
            (df_auctions["time_offset"] >= 0) &
            (df_auctions["time_offset"] <= max_hours_back)
        ].copy()
        df_item_full = df_item_full[df_item_full["id"] != auction_id]

        # Fresh relist row at present target price (reuse the same auction id)
        relist_row = cheapest_row.copy()
        relist_row["id"] = auction_id
        relist_row["buyout"] = float(target_price)
        relist_row["bid"] = 0.0
        relist_row["time_left"] = 48.0
        relist_row["current_hours"] = 0.0
        relist_row["snapshot_time"] = prediction_time
        relist_row["time_offset"] = 0
        relist_row["hour_of_week"] = prediction_time.weekday() * 24 + prediction_time.hour

        # Append and predict with full context
        df_item_full = pd.concat([df_item_full, pd.DataFrame([relist_row])], ignore_index=True)

        prediction_df = predict_dataframe(
            model=model,
            df_auctions=df_item_full,
            prediction_time=prediction_time,
            feature_stats=feature_stats,
            max_hours_back=max_hours_back,
            max_sequence_length=max_sequence_length
        )
        if prediction_df.empty:
            continue

        # Select the prediction for our relisted auction by id (unique after the drop)
        pred_row = prediction_df[prediction_df["id"] == auction_id].iloc[0]

        if pred_row["prediction_q50"] >= sold_threshold:
            continue

        recommendation = pd.Series({
            "item_id": cheapest_row.get("item_id"),
            "buyout": round(cheapest_buyout, 2),
            "suggested_price": round(target_price, 2),
            "quantity": pred_row.get("quantity", cheapest_row.get("quantity")),
            "bonus_lists": pred_row.get("bonus_lists", relist_row.get("bonus_lists")),
            "modifier_types": pred_row.get("modifier_types", relist_row.get("modifier_types")),
            "modifier_values": pred_row.get("modifier_values", relist_row.get("modifier_values")),
            "prediction_q10": float(pred_row["prediction_q10"]),
            "prediction_q50": float(pred_row["prediction_q50"]),
            "prediction_q90": float(pred_row["prediction_q90"]),
            "is_short_duration": float(pred_row["is_short_duration"]),
            "potential_profit": round(potential_profit, 2),
        })
        recommendations_list.append(recommendation)

    if not recommendations_list:
        return pd.DataFrame()

    return pd.DataFrame(recommendations_list).sort_values(
        ["potential_profit", "prediction_q50"],
        ascending=[False, False]
    )

def generate_recommendations_ui(expected_profit, threshold):
    """UI function for generating recommendations"""
    global recommendations, sold_threshold
    
    expected_profit = float(expected_profit)
    sold_threshold = float(threshold)
    
    recommendations = generate_recommendations(expected_profit)
    
    if recommendations.empty:
        return "No recommendations found matching your criteria. Try lowering the expected profit.", pd.DataFrame()
    else:
        return f"Found {len(recommendations)} recommendations!", recommendations

def create_ui():
    """Create the Gradio interface"""
    with gr.Blocks(title="🎮 Auction House Data 💰", theme=gr.themes.Default()) as app:
        gr.Markdown("# Auction House Data")
        
        with gr.Tab("Current Auctions"):
            gr.Markdown("## Current Auctions")
            
            with gr.Row():
                item_search = gr.Textbox(label="Search by Item ID", placeholder="Enter item ID...")
                search_button = gr.Button("Search")
                time_offset_filter = gr.Slider(label="Display Time Offset (hours ago)", minimum=0, maximum=48, value=0, step=1)
                predict_button = gr.Button("Predict Sale Probability")
                store_csv_button = gr.Button("Store CSV")
            
            display_columns = ['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_lists', 'modifier_types', 'modifier_values', 'current_hours', 'time_offset']

            initial_display = pd.DataFrame()
            if df_auctions is not None:
                current_slice = df_auctions[df_auctions['time_offset'] == 0]
                initial_display = current_slice[display_columns].head(100)

            auctions_display = gr.Dataframe(initial_display, interactive=False)
            store_csv_status = gr.Markdown("")
            
            def filter_auctions(item_id, time_offset_value):
                if df_auctions is None:
                    return pd.DataFrame()

                item_id = (item_id or "").strip()
                time_offset_value = float(time_offset_value or 0)
                filtered_df = df_auctions[df_auctions['time_offset'] == time_offset_value]
                if item_id:
                    filtered_df = filtered_df[filtered_df['item_id'] == item_id]

                return filtered_df[display_columns].head(100)
            
            def predict_filtered_auctions(item_id, time_offset_value):
                global model, feature_stats, prediction_time

                if df_auctions is None:
                    return pd.DataFrame()

                item_id = (item_id or "").strip()
                time_offset_value = float(time_offset_value or 0)

                if not item_id:
                    filtered_df = df_auctions.head(100).copy()
                else:
                    filtered_df = df_auctions[df_auctions['item_id'] == item_id].copy()
                
                if filtered_df.empty:
                    return pd.DataFrame()

                prediction_results = predict_dataframe(
                    model, 
                    filtered_df, 
                    prediction_time, 
                    feature_stats,
                    max_hours_back=max_hours_back,
                    max_sequence_length=max_sequence_length
                )
                
                prediction_results = prediction_results[prediction_results['time_offset'] == 0]

                if prediction_results.empty:
                    return pd.DataFrame(columns=['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_lists', 'modifier_types', 'modifier_values', 'current_hours', 'prediction_q10', 'prediction_q50', 'prediction_q90', 'is_short_duration'])

                display_columns_prediction = display_columns + ['prediction_q10', 'prediction_q50', 'prediction_q90', 'is_short_duration']
                return prediction_results[display_columns_prediction].head(100)

            def store_item_csv(item_id, _time_offset_value):
                if df_auctions is None:
                    return "No auctions loaded yet."

                item_id = (item_id or "").strip()
                if not item_id:
                    return "Please provide an item ID before storing."

                item_df = df_auctions[df_auctions['item_id'] == item_id].copy()
                if item_df.empty:
                    return f"No rows found for item {item_id}."

                item_df = item_df.sort_values("current_hours")

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
            gr.Markdown("This tab recommends items to flip using their **25th percentile (q25) price * 0.7** as the suggested selling price.")
            gr.Markdown("Only items with potential profit >= your minimum expected profit filter will be shown.")
            gr.Markdown("**Prediction columns explained:**")
            gr.Markdown("- **prediction_q10**: 10th percentile prediction (pessimistic - 90% chance it takes longer)")
            gr.Markdown("- **prediction_q50**: 50th percentile prediction (median)")
            gr.Markdown("- **prediction_q90**: 90th percentile prediction (optimistic - 90% chance it sells faster)")
            gr.Markdown("- **is_short_duration**: Probability (0-1) that the item will sell in less than 8 hours")
            gr.Markdown(f"Items are considered likely to sell if prediction_q50 < {sold_threshold} hours")
            
            with gr.Row():
                expected_profit = gr.Number(label="Minimum Expected Profit Filter (gold)", value=100, minimum=0, step=100)
                sold_threshold_input = gr.Slider(label="Sale Time Threshold (hours)", minimum=1, maximum=24, value=4, step=1)
        
            generate_button = gr.Button("Generate Recommendations")
            result_text = gr.Markdown()
            recommendations_output = gr.Dataframe()
            
            generate_button.click(
                generate_recommendations_ui, 
                inputs=[expected_profit, sold_threshold_input], 
                outputs=[result_text, recommendations_output]
            )
        
        with gr.Tab("Individual Flipping"):
            gr.Markdown("## Individual Item Flipping Recommendations")
            gr.Markdown("**Prediction columns explained:**")
            gr.Markdown("- **predicted_hours_q10**: 10th percentile (optimistic timing)")
            gr.Markdown("- **predicted_hours_q50**: 50th percentile (median hours to sale)")
            gr.Markdown("- **predicted_hours_q90**: 90th percentile (pessimistic timing)")
            gr.Markdown("- **is_short_duration**: Probability (0-1) that the item will sell in less than 8 hours")
            
            with gr.Row():
                flip_item_search = gr.Textbox(label="Search by Item ID", placeholder="Enter item ID...")
                flip_search_button = gr.Button("Search")
            
            item_auctions_display = gr.Dataframe(interactive=False)
            
            with gr.Row():
                custom_buyout = gr.Number(label="Custom Buyout Price (gold)", value=0, minimum=0, step=100)
                predict_flip_button = gr.Button("Predict Flip")
            
            flip_result = gr.Dataframe(interactive=False)
            
            def search_item_for_flip(item_id):
                global df_auctions

                if not model_loaded:
                    load_data_and_model()

                item_id = (item_id or "").strip()
                if not item_id or df_auctions is None:
                    return pd.DataFrame()
                
                filtered_df = df_auctions[
                    (df_auctions['item_id'] == item_id) &
                    (df_auctions['time_offset'] == 0)
                ]
                display_columns = ['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_lists', 'modifier_types', 'modifier_values']
                return filtered_df[display_columns]            
            def predict_item_flip(item_id, custom_price):
                global model, feature_stats, prediction_time, max_hours_back, df_auctions

                if not model_loaded:
                    load_data_and_model()

                item_id = (item_id or "").strip()
                if not item_id or df_auctions is None:
                    return pd.DataFrame()

                item_df = df_auctions[df_auctions['item_id'] == item_id].copy()
                
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

                df_item_full = df_auctions[
                    (df_auctions['item_index'] == item_index) &
                    (df_auctions['time_offset'] >= 0) &
                    (df_auctions['time_offset'] <= max_hours_back)
                ].copy()

                df_item_full = df_item_full[df_item_full['id'] != auction_id]

                # New relist row mirrors recommendation flow
                relist_row = lowest_auction.copy()
                relist_row['buyout'] = float(target_price)
                relist_row['bid'] = 0.0
                relist_row['time_left'] = 48.0
                relist_row['current_hours'] = 0.0
                relist_row['snapshot_time'] = prediction_time
                relist_row['time_offset'] = 0
                relist_row['hour_of_week'] = prediction_time.weekday() * 24 + prediction_time.hour

                df_item_full = pd.concat([df_item_full, pd.DataFrame([relist_row])], ignore_index=True)

                prediction_df = predict_dataframe(
                    model, 
                    df_item_full, 
                    prediction_time, 
                    feature_stats,
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
                    'is_short_duration': flip_prediction['is_short_duration']
                }])
                
                return result            
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
    
    app = create_ui()
    app.launch(share=True)
