import torch
import sys
import os
import pandas as pd
import json
import numpy as np
import gradio as gr

from datetime import datetime, timedelta
from pathlib import Path

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from tqdm import tqdm
from src.models.inference import predict_dataframe
from src.data.utils import load_auctions_from_sample
from src.models.auction_transformer import AuctionTransformer

model_loaded = False
df_auctions = None
historical_prices = None
model = None
feature_stats = None
prediction_time = None
recommendations = None

def load_data_and_model():
    """Load data and model only once"""
    global model_loaded, df_auctions, model, feature_stats, prediction_time, historical_prices
    
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
    prediction_time = prediction_time.replace(minute=0, second=0, microsecond=0)

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
        os.path.join(base_path, 'models/auction_transformer_40M_39GB/last.ckpt'),
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
        last_days=7
    )
    df_auctions['item_id'] = df_auctions['item_index'].map(idx_to_item)

    print(f'Number of auctions loaded: {len(df_auctions)}')

    historical_prices = df_auctions.groupby('item_index')['buyout'].median().reset_index()
    historical_prices.columns = ['item_index', 'historical_price']
    
    model_loaded = True
    return "Model and data loaded successfully!"

def generate_recommendations(expected_profit, min_sale_probability):
    """Generate auction recommendations based on expected profit and minimum sale probability."""
    global df_auctions, model, feature_stats, prediction_time
    
    if not model_loaded:
        load_data_and_model()
    
    recommendations = []
    unique_items = df_auctions['item_index'].unique()
    
    for item_idx in tqdm(unique_items):
        item_df = df_auctions[df_auctions['item_index'] == item_idx].copy()
        
        if item_df.empty:
            continue
            
        # Get the auction with the lowest buyout price
        min_buyout_idx = item_df['buyout'].idxmin()
        lowest_auction = item_df.loc[min_buyout_idx].copy()
        
        # Create a modified version of the item_df with our modified auction
        modified_item_df = item_df.copy()
        modified_item_df.loc[min_buyout_idx, 'buyout'] = lowest_auction['buyout'] + expected_profit
        modified_item_df.loc[min_buyout_idx, 'bid'] = 0
        modified_item_df.loc[min_buyout_idx, 'time_left'] = 48.0
        modified_item_df.loc[min_buyout_idx, 'current_hours'] = 0
        
        # Predict sale probability for all auctions of this item type
        prediction_df = predict_dataframe(
            model, 
            modified_item_df, 
            prediction_time, 
            feature_stats
        )

        if prediction_df.empty:
            continue
        
        modified_auction_prediction = prediction_df.loc[min_buyout_idx]
        if modified_auction_prediction['sale_probability'] >= min_sale_probability:
            recommendation = pd.Series({
                'item_id': lowest_auction['item_id'],
                'buyout': modified_auction_prediction['buyout'] - expected_profit,
                'suggested_price': lowest_auction['buyout'] + expected_profit,
                'quantity': modified_auction_prediction['quantity'],
                'bonus_lists': modified_auction_prediction['bonus_lists'],
                'modifier_types': modified_auction_prediction['modifier_types'], 
                'modifier_values': modified_auction_prediction['modifier_values'],
                'prediction': modified_auction_prediction['prediction'],
                'sale_probability': modified_auction_prediction['sale_probability'],
                'original_price': lowest_auction['buyout']
            })
            recommendations.append(recommendation)

    if recommendations:
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('sale_probability', ascending=False)
        return recommendations_df
    else:
        return pd.DataFrame()

def generate_historical_price_recommendations(min_sale_probability):
    """Generate auction recommendations based on historical prices and minimum sale probability."""
    global df_auctions, model, feature_stats, prediction_time, historical_prices
    
    if not model_loaded:
        load_data_and_model()
    
    recommendations = []
    unique_items = df_auctions['item_index'].unique()
    
    for item_idx in tqdm(unique_items):
        item_df = df_auctions[df_auctions['item_index'] == item_idx].copy()
        
        if item_df.empty:
            continue
            
        # Get the auction with the lowest buyout price
        min_buyout_idx = item_df['buyout'].idxmin()
        lowest_auction = item_df.loc[min_buyout_idx].copy()
        
        # Get historical price for this item
        hist_price = historical_prices[historical_prices['item_index'] == item_idx]['historical_price'].values
        if len(hist_price) == 0:
            print(f"No historical price found for item {item_idx}")
            continue
        
        historical_price = hist_price[0]
        
        # Skip if historical price is lower than current lowest price
        if lowest_auction['buyout'] >= historical_price * 0.9:
            continue

        # Create a modified version of the item_df with our modified auction
        modified_item_df = item_df.copy()
        modified_item_df.loc[min_buyout_idx, 'buyout'] = historical_price
        modified_item_df.loc[min_buyout_idx, 'bid'] = 0
        modified_item_df.loc[min_buyout_idx, 'time_left'] = 48.0
        modified_item_df.loc[min_buyout_idx, 'current_hours'] = 0
        
        # Predict sale probability for all auctions of this item type
        prediction_df = predict_dataframe(
            model, 
            modified_item_df, 
            prediction_time, 
            feature_stats
        )

        if prediction_df.empty:
            continue
        
        modified_auction_prediction = prediction_df.loc[min_buyout_idx]
        if modified_auction_prediction['sale_probability'] >= min_sale_probability:
            recommendation = pd.Series({
                'item_id': lowest_auction['item_id'],
                'buyout': lowest_auction['buyout'],
                'suggested_price': historical_price,
                'quantity': modified_auction_prediction['quantity'],
                'bonus_lists': modified_auction_prediction['bonus_lists'],
                'modifier_types': modified_auction_prediction['modifier_types'], 
                'modifier_values': modified_auction_prediction['modifier_values'],
                'prediction': modified_auction_prediction['prediction'],
                'sale_probability': modified_auction_prediction['sale_probability'],
                'potential_profit': historical_price - lowest_auction['buyout']
            })
            recommendations.append(recommendation)

    if recommendations:
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('potential_profit', ascending=False)
        return recommendations_df
    else:
        return pd.DataFrame()

def generate_recommendations_ui(expected_profit, min_sale_probability):
    """UI function for generating recommendations"""
    global recommendations
    
    expected_profit = float(expected_profit)
    min_sale_probability = float(min_sale_probability)
    
    recommendations = generate_recommendations(expected_profit, min_sale_probability)
    
    if recommendations.empty:
        return "No recommendations found matching your criteria. Try lowering the minimum sale probability or expected profit."
    else:
        return f"Found {len(recommendations)} recommendations!", recommendations

def generate_historical_recommendations_ui(min_sale_probability):
    """UI function for generating historical price recommendations"""
    
    min_sale_probability = float(min_sale_probability)
    
    recommendations = generate_historical_price_recommendations(min_sale_probability)
    
    if recommendations.empty:
        return "No recommendations found matching your criteria. Try lowering the minimum sale probability.", pd.DataFrame()
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
                predict_button = gr.Button("Predict Sale Probability")
            
            auctions_display = gr.Dataframe(df_auctions.head(100), interactive=False)
            
            def filter_auctions(item_id):
                if not item_id:
                    return df_auctions.head(100)
                
                filtered_df = df_auctions[df_auctions['item_id'] == item_id]
                display_columns = ['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_lists', 'modifier_types', 'modifier_values', 'first_appearance', 'last_appearance', 'current_hours']
                return filtered_df[display_columns]
            
            def predict_filtered_auctions(item_id):
                global model, feature_stats, prediction_time

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
                    feature_stats
                )
                
                display_columns = ['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_lists', 'modifier_types', 'modifier_values', 'first_appearance', 'last_appearance', 'current_hours', 'prediction', 'sale_probability']
                return prediction_results[display_columns]
            
            search_button.click(
                filter_auctions,
                inputs=[item_search],
                outputs=[auctions_display]
            )
            
            predict_button.click(
                predict_filtered_auctions,
                inputs=[item_search],
                outputs=[auctions_display]
            )
        
        with gr.Tab("Recommendations"):
            gr.Markdown("## Auction Recommendations")
            
            with gr.Row():
                expected_profit = gr.Number(label="Expected Profit (gold)", value=100, minimum=0, step=100)
                min_sale_probability = gr.Slider(label="Minimum Sale Probability", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
            
            generate_button = gr.Button("Generate Recommendations")
            result_text = gr.Markdown()
            recommendations_output = gr.Dataframe()
            
            generate_button.click(
                generate_recommendations_ui, 
                inputs=[expected_profit, min_sale_probability], 
                outputs=[result_text, recommendations_output]
            )
        
        with gr.Tab("Individual Flipping"):
            gr.Markdown("## Individual Item Flipping Recommendations")
            
            with gr.Row():
                flip_item_search = gr.Textbox(label="Search by Item ID", placeholder="Enter item ID...")
                flip_search_button = gr.Button("Search")
            
            item_auctions_display = gr.Dataframe(interactive=False)
            
            with gr.Row():
                custom_buyout = gr.Number(label="Custom Buyout Price (gold)", value=0, minimum=0, step=100)
                predict_flip_button = gr.Button("Predict Flip")
            
            flip_result = gr.Dataframe(interactive=False)
            
            def search_item_for_flip(item_id):
                if not item_id or not model_loaded:
                    return pd.DataFrame()
                
                filtered_df = df_auctions[df_auctions['item_id'] == item_id]
                display_columns = ['item_id', 'buyout', 'quantity', 'time_left', 'context', 'bonus_lists', 'modifier_types', 'modifier_values']
                return filtered_df[display_columns]
            
            def predict_item_flip(item_id, custom_price):
                global model, feature_stats, prediction_time
                
                if not item_id or not model_loaded:
                    return pd.DataFrame()
                
                item_df = df_auctions[df_auctions['item_id'] == item_id].copy()
                
                if item_df.empty:
                    print("No auctions found for this item")
                    return pd.DataFrame()
                
                min_buyout_idx = item_df['buyout'].idxmin()
                lowest_auction = item_df.loc[min_buyout_idx].copy()
                
                modified_item_df = item_df.copy()
                modified_item_df.loc[min_buyout_idx, 'buyout'] = custom_price
                modified_item_df.loc[min_buyout_idx, 'bid'] = 0
                modified_item_df.loc[min_buyout_idx, 'time_left'] = 48.0
                modified_item_df.loc[min_buyout_idx, 'current_hours'] = 0
                
                prediction_df = predict_dataframe(
                    model, 
                    modified_item_df, 
                    prediction_time, 
                    feature_stats
                )

                print(prediction_df)
                
                if prediction_df.empty:
                    print("No prediction found for this item")
                    return pd.DataFrame()
                
                flip_prediction = prediction_df.loc[min_buyout_idx]
                
                result = pd.DataFrame([{
                    'item_id': lowest_auction['item_id'],
                    'quantity': lowest_auction['quantity'],
                    'original_price': lowest_auction['buyout'],
                    'custom_price': custom_price,
                    'potential_profit': custom_price - lowest_auction['buyout'],
                    'predicted_hours_to_sale': flip_prediction['prediction'],
                    'sale_probability': flip_prediction['sale_probability'],
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
        
        with gr.Tab("Historical Price Flipping"):
            gr.Markdown("## Historical Price Flipping Recommendations")
            gr.Markdown("This tab recommends items to flip based on their historical median prices from the last 7 days.")
            
            with gr.Row():
                hist_min_sale_probability = gr.Slider(label="Minimum Sale Probability", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
            
            hist_generate_button = gr.Button("Generate Historical Price Recommendations")
            hist_result_text = gr.Markdown()
            hist_recommendations_output = gr.Dataframe()
            
            hist_generate_button.click(
                generate_historical_recommendations_ui, 
                inputs=[hist_min_sale_probability], 
                outputs=[hist_result_text, hist_recommendations_output]
            )
    
    return app

if __name__ == "__main__":
    load_data_and_model()
    
    app = create_ui()
    app.launch(share=True)
