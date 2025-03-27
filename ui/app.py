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

time_left_mapping = {
    'VERY_LONG': 48,
    'LONG': 12,
    'MEDIUM': 2,
    'SHORT': 0.5
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
