import torch
import sys
import os
import gradio as gr
import pandas as pd
import json
import pickle
import numpy as np
import time

from datetime import datetime, timedelta
from pathlib import Path

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from datetime import datetime
from scripts.utils import get_current_auctions
from tqdm import tqdm
from src.models.auction_rnn import AuctionPredictor
from src.models.inference import predict_dataframe

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lambda_value = 0.0401

with open(os.path.join(base_path, "scripts", "config.json")) as json_data:
    config = json.load(json_data)

time_left_options = [12, 24, 48]

def load_auctions():
    if not os.path.exists('auctions.pkl'):
        print("No saved auction data found, fetching new data...")
        auction_data = get_current_auctions(config)
        data_with_timestamp = {
            'timestamp': datetime.now(),
            'data': auction_data
        }
        with open('auctions.pkl', 'wb') as f:
            pickle.dump(data_with_timestamp, f)
        return auction_data
    
    with open('auctions.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    current_time = datetime.now()
    time_difference = current_time - saved_data['timestamp']
    
    if time_difference > timedelta(hours=1):
        print("Saved auction data is older than 1 hour, fetching new data...")
        auction_data = get_current_auctions(config)
        data_with_timestamp = {
            'timestamp': current_time,
            'data': auction_data
        }
        with open('auctions.pkl', 'wb') as f:
            pickle.dump(data_with_timestamp, f)
        return auction_data
    
    return saved_data['data']

auction_data = load_auctions()
df = pd.DataFrame(auction_data, columns=['id', 'item_id', 'bid', 'buyout', 'quantity', 'time_left', 'rand', 'seed'])
df = df.drop(columns=['rand', 'seed'])

items_df = pd.read_csv(os.path.join(base_path, "data", "items.csv"))

current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
df['first_appearance_timestamp'] = current_timestamp
df['unit_price'] = df['buyout'] / df['quantity']

df = df.merge(items_df, on='item_id', how='inner')
df = df.drop(columns=['sell_price_silver', 'purchase_price_silver'], errors='ignore')

print("Data loaded and prepared successfully.")

n_items = len(items_df)
item_to_index = {item_id: i + 2 for i, item_id in enumerate(items_df['item_id'])}
item_to_index[0] = 0 
item_to_index[1] = 1  
print(f"Number of unique items: {n_items}")

time_left_mapping = {
    'VERY_LONG': 48,
    'LONG': 12,
    'MEDIUM': 2,
    'SHORT': 0.5
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AuctionPredictor(
    n_items=len(item_to_index),             
    input_size=5,                   
    encoder_hidden_size=2048,
    decoder_hidden_size=2048,
    num_layers=3,
    item_index=3,                   
    embedding_size=1024,
    dropout_p=0.2,
    bidirectional=False
).to(device)

print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')

model_path = os.path.join(base_path, "models", "checkpoint_epoch_0_iter_51115.pt")
checkpoint = torch.load(model_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  
print('Pre-trained RNN model loaded successfully.')

def compute_auctions_appearances(data_dir):
    print("Computing auction appearances...")

    file_info = {}
    auction_appearances = {}

    for root, dirs, files in os.walk(data_dir):
        for filename in tqdm(files):
            filepath = os.path.join(root, filename)
            date = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')
            file_info[filepath] = date

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}
    
    all_auctions = []
    
    for filepath in list(file_info.keys()):
        with open(filepath, 'r') as f:
            try:
                json_data = json.load(f)
                
                if 'auctions' not in json_data:
                    print(f"File {filepath} does not contain 'auctions' key, skipping.")
                    continue
                
                auction_data = json_data['auctions']
                timestamp = file_info[filepath]
                
                for auction in auction_data:
                    auction_id = int(auction['id'])
                    auction['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    
                    if auction_id not in auction_appearances:
                        auction_appearances[auction_id] = {'first': timestamp, 'last': timestamp}
                    else:
                        auction_appearances[auction_id]['last'] = timestamp
                
                all_auctions.extend(auction_data)
            except json.JSONDecodeError as e:
                print(f"Error loading file {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error loading file {filepath}: {e}")
                continue

    print("Auction appearances computed successfully.")
    return auction_appearances

auction_appearances = compute_auctions_appearances(os.path.join(base_path, "ui", "sample"))

def predict_item_sale(realm_id, faction, item_id, quantity, buyout, bid, time_left):
    raise NotImplementedError("This function hasn't been implemented yet")

    item_id = int(item_id)
    item_auctions = df[df['item_id'] == item_id]

    single_item_df = pd.DataFrame({
        'id': [0],
        'item_id': [item_id],
        'bid': [bid],
        'buyout': [buyout],
        'quantity': [quantity],
        'time_left': time_left,
        'hours_since_first_appearance': [0],
    })

    single_item_df['item_id'] = single_item_df['item_id'].astype(int)
    items_df['item_id'] = items_df['item_id'].astype(int)

    single_item_df = single_item_df.merge(items_df, on='item_id', how='left')
    merged_df = pd.concat([item_auctions, single_item_df], ignore_index=True)

    return f"Your item has a sale probability of {sale_probability * 100:.2f}%. Estimated sale time {prediction:.2f} hours."
    

def get_suggested_items(realm_id, faction, item_class, item_subclass, desired_profit, sale_probability, price_filter_max):
    suggested_items = df.copy()

    if item_class:
        suggested_items = suggested_items[(suggested_items['item_class'] == item_class)]
    if item_subclass:
        suggested_items = suggested_items[(suggested_items['item_subclass'] == item_subclass)]

    print(f"Number of items after filtering: {len(suggested_items)}")

    prediction_time = datetime.now()
    prediction_time = prediction_time.replace(minute=0, second=0, microsecond=0)

    suggested_items['first_appearance'] = suggested_items['id'].apply(lambda x: auction_appearances[int(x)]['first'] if x in auction_appearances else prediction_time)
    suggested_items['hours_since_first_appearance'] = (prediction_time - pd.to_datetime(suggested_items['first_appearance'])).dt.total_seconds() / 3600

    print("Predicting sale times for items...")
    suggested_items = predict_dataframe(model, suggested_items, prediction_time, time_left_mapping, item_to_index, lambda_value, device)    

    if price_filter_max:
        suggested_items = suggested_items[suggested_items['buyout'] <= price_filter_max]

    if sale_probability:
        suggested_items = suggested_items[suggested_items['sale_probability'] >= sale_probability]

    suggested_items = suggested_items.sort_values(by='sale_probability', ascending=False)

    return suggested_items[['item_name', 'item_class', 'item_subclass', 'bid', 'buyout', 'quantity', 'time_left', 'first_appearance', 'hours_since_first_appearance', 'prediction', 'sale_probability']]


with gr.Blocks(title="Auction Tools") as demo:
    gr.Markdown("## Maximize your auction profits!")

    with gr.Tab("Sale Prediction"):
        gr.Markdown("### Predict when your item will sell")
        realm_id_input = gr.Textbox(label="Realm ID", value=str(config["realm_id"]))
        faction_input = gr.Radio(["Alliance", "Horde"], label="Faction", value="Alliance")
        item_input = gr.Textbox(label="Item ID")
        quantity_input = gr.Number(label="Quantity")
        buyout_input = gr.Number(label="Buyout")
        bid_input = gr.Number(label="Bid")
        time_left_input = gr.Dropdown(time_left_options, label="Time Left")
        predict_button = gr.Button("Predict")
        prediction_output = gr.Textbox(label="AI Prediction")
        predict_button.click(predict_item_sale, inputs=[realm_id_input, faction_input, item_input, quantity_input, buyout_input, bid_input, time_left_input], outputs=prediction_output)

    with gr.Tab("Resale Suggestions"):
        gr.Markdown("### Discover resale opportunities")
        realm_id_input = gr.Textbox(label="Realm ID", value=str(config["realm_id"]))
        faction_input = gr.Radio(["Alliance", "Horde"], label="Faction", value="Horde")
        item_class_input = gr.Dropdown(choices=items_df['item_class'].unique().tolist(), label="Item Class", multiselect=False)
        item_subclass_input = gr.Dropdown(choices=items_df['item_subclass'].unique().tolist(), label="Item Subclass", multiselect=False)
        desired_profit_input = gr.Number(label="Desired Profit (Gold)")
        sale_probability_input = gr.Slider(label="Minimum Sale Probability (%)", minimum=0, maximum=100, step=1, value=80)
        price_filter_max_input = gr.Number(label="Maximum Original Buyout Price (Gold)", value=500)
        suggest_button = gr.Button("Suggest Items")
        suggested_items_output = gr.DataFrame(label="Suggested Items", interactive=False, height=300)
        suggest_button.click(lambda realm, faction, item_class, item_subclass, profit, sale_probability, price_filter_max: get_suggested_items(realm, faction, item_class, item_subclass, profit, sale_probability / 100.0, price_filter_max), inputs=[realm_id_input, faction_input, item_class_input, item_subclass_input, desired_profit_input, sale_probability_input, price_filter_max_input], outputs=suggested_items_output)

demo.launch(share=False)
