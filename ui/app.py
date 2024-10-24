import torch
import sys
import os
import gradio as gr
import pandas as pd
import json
import pickle
import numpy as np
import time

from pathlib import Path

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from datetime import datetime
from scripts.utils import get_current_auctions
from tqdm import tqdm
from prediction_engine.model import AuctionPredictor

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lambda_value = 0.0401

with open(os.path.join(base_path, "scripts", "config.json")) as json_data:
    config = json.load(json_data)

time_left_options = [12, 24, 48]

if not os.path.exists('auctions.pkl'):
    auction_data = get_current_auctions(config)
    with open('auctions.pkl', 'wb') as f:
        pickle.dump(auction_data, f)
else:
    with open('auctions.pkl', 'rb') as f:
        auction_data = pickle.load(f)

df = pd.DataFrame(auction_data, columns=['auction_id', 'item_id', 'bid', 'buyout', 'quantity', 'time_left', 'rand', 'seed'])
df = df.drop(columns=['rand', 'seed'])

items_df = pd.read_csv(os.path.join(base_path, "data", "items.csv"))

current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
df['first_appearance_timestamp'] = current_timestamp
df['unit_price'] = df['buyout'] / df['quantity']

df = df.merge(items_df, on='item_id', how='inner')
df = df.drop(columns=['sell_price_silver', 'purchase_price_silver'], errors='ignore')

df.to_csv(os.path.join(base_path, "ui", "auctions.csv"), index=False)

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
device = 'cpu'

model = AuctionPredictor(
    n_items=len(item_to_index),             
    input_size=5,                   
    encoder_hidden_size=1024,
    decoder_hidden_size=1024,
    item_index=3,                   
    embedding_size=512,
    dropout_p=0.2,
    bidirectional=False
).to(device)

print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')

model_path = os.path.join(base_path, "models", "checkpoint_epoch_1_iter_10336.pt")
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

def get_suggested_items(realm_id, faction, item_class, item_subclass, desired_profit, sale_probability, price_filter_max):
    suggested_items = df.copy()

    if item_class:
        suggested_items = suggested_items[(suggested_items['item_class'] == item_class)]
    if item_subclass:
        suggested_items = suggested_items[(suggested_items['item_subclass'] == item_subclass)]

    print(f"Number of items after filtering: {len(suggested_items)}")

    prediction_time = datetime.now()
    prediction_time = prediction_time.replace(minute=0, second=0, microsecond=0)

    suggested_items['first_appearance'] = suggested_items['auction_id'].apply(lambda x: auction_appearances[int(x)]['first'] if x in auction_appearances else prediction_time)
    suggested_items['hours_since_first_appearance'] = (prediction_time - pd.to_datetime(suggested_items['first_appearance'])).dt.total_seconds() / 3600

    auctions_by_item = {}

    for item_id in tqdm(suggested_items['item_id'].unique()):
        for index, auction in suggested_items[suggested_items['item_id'] == item_id].iterrows():
            auction_id = auction['auction_id']
            item_id = auction['item_id']
            time_left_numeric = time_left_mapping.get(auction['time_left'], 0) / 48.0
            bid = np.log1p(auction['bid'] / 10000.0) / 15.0
            buyout = np.log1p(auction['buyout'] / 10000.0) / 15.0
            quantity = auction['quantity'] / 200.0
            item_index = item_to_index.get(item_id, 1)
            hours_since_first_appearance = auction['hours_since_first_appearance'] / 48.0

            processed_auction = [
                auction_id,
                bid, 
                buyout,  
                quantity, 
                item_index,
                time_left_numeric, 
                hours_since_first_appearance
            ]
            
            if item_index not in auctions_by_item:
                auctions_by_item[item_index] = []

            auctions_by_item[item_index].append(processed_auction)

    for item_index in auctions_by_item:
        auctions_by_item[item_index] = np.array(auctions_by_item[item_index])
        
        X = torch.tensor(auctions_by_item[item_index][:, 1:], dtype=torch.float32)
        lengths = torch.tensor([len(auctions_by_item[item_index])])
        y = model(X.unsqueeze(0), lengths)

        for i, auction_id in enumerate(auctions_by_item[item_index][:, 0]):
            suggested_items.loc[suggested_items['auction_id'] == auction_id, 'prediction'] = round(y[0][i].item(), 2)
            suggested_items.loc[suggested_items['auction_id'] == auction_id, 'sale_probability'] = np.exp(-lambda_value * y[0][i].item())

    # Filter out items with buyout price greater than price_filter_max
    if price_filter_max:
        suggested_items = suggested_items[suggested_items['buyout'] <= price_filter_max]

    # Filter out items with probability of sale less than sale_probability
    # if sale_probability:
    #    suggested_items = suggested_items[suggested_items['probability_of_sale'] >= sale_probability]

    return suggested_items[['item_name', 'item_class', 'item_subclass', 'bid', 'buyout', 'quantity', 'time_left', 'hours_since_first_appearance', 'prediction', 'probability_of_sale']]


with gr.Blocks(title="Auction Tools") as demo:
    gr.Markdown("## Maximize your auction profits!")

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
