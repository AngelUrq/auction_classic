import gradio as gr
import pandas as pd
import json
import pickle
from datetime import datetime
from utils import get_auction_data
from transformers import add_features, transform_data
import numpy as np

config = json.load(open("config.json", "r"))
model = pickle.load(open('forest_model.pkl', 'rb'))
items_df = pd.read_csv('data/items.csv')
time_left_options = [12, 24, 48]

def load_and_prepare_data(item_data=None):
    try:
        data = get_auction_data()
        df = pd.DataFrame(data, columns=['auction_id', 'item_id', 'bid_in_gold', 'buyout_in_gold', 'quantity', 'time_left', 'rand', 'seed'])
        df = df.drop(columns=['rand', 'seed'])

        if item_data:
            item_df = pd.DataFrame([item_data])
            df = pd.concat([df, item_df], ignore_index=True)
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['first_appearance_timestamp'] = current_timestamp
        df['first_appearance_year'] = int(current_timestamp[:4])
        df['first_appearance_month'] = int(current_timestamp[5:7])
        df['first_appearance_day'] = int(current_timestamp[8:10])
        df['first_appearance_hour'] = int(current_timestamp[11:13])
        df['hours_on_sale'] = 0
        df['unit_price'] = df['buyout_in_gold'] / df['quantity']

        df = df.merge(items_df, on='item_id', how='inner')
        df = df.drop(columns=['bid', 'buyout', 'max_count', 'sell_price_silver', 'purchase_price_silver'], errors='ignore')

        df = add_features(df)
        X, _ = transform_data(df)
        df['prediction'] = model.predict(X)
        return df
    except Exception as e:
        print(f"Error loading and preparing data: {e}")
        return None
    
auction_data = load_and_prepare_data()
def predict_item_sale(realm_id, faction, item, quantity, buyout, bid, time_left_hours):
    if not all([realm_id, faction, item, quantity, buyout, bid, time_left_hours]):
        return "Error: All fields are required and must contain valid data."
    if quantity <= 0 or buyout <= 0:
        return "Error: Quantity and buyout price must be greater than zero."
    try:
        item_id = int(item)
    except ValueError:
        return "Error: The item ID must be an integer."
    if item_id not in items_df['item_id'].values:
        return "Error: Item not found. Please check the data."
    if time_left_hours not in time_left_options:
        return "Error: Remaining time must be 12, 24, or 48 hours."

    item_data = {
        'auction_id': 0,
        'item_id': item_id,
        'quantity': quantity,
        'buyout_in_gold': buyout,
        'bid_in_gold': bid,
        'time_left': time_left_hours
    }

    df = load_and_prepare_data(item_data)
    if df is None:
        return "Error: Could not retrieve auction data."
    user_prediction = df['prediction'].iloc[-1]
    if user_prediction < 8:
        return f"Your item will sell in approximately {user_prediction:.2f} hours."
    else:
        return f"Your item may take more than {user_prediction:.2f} hours to sell."

def get_suggested_items(realm_id, faction, desired_profit):
    if auction_data is not None:
        auction_data['predicted_resale_price'] = auction_data['unit_price'] * (1 + desired_profit / 100)
        auction_data['profit_margin'] = auction_data['predicted_resale_price'] - auction_data['buyout_in_gold']
        suggested_items = auction_data[auction_data['profit_margin'] > 0]
        return suggested_items[['item_name', 'item_class', 'unit_price', 'buyout_in_gold', 'predicted_resale_price', 'profit_margin', 'prediction']]
    else:
        return pd.DataFrame()
def get_quick_selling_items(budget):
    df = load_and_prepare_data()
    if df is not None:
        quick_selling_items = df[(df['prediction'] < 8) & (df['buyout_in_gold'] <= budget)]
        return quick_selling_items[['item_id', 'item_name', 'quantity', 'buyout_in_gold', 'bid_in_gold', 'time_left', 'prediction']]
    else:
        return pd.DataFrame()

with gr.Blocks(title="Auction Tools") as demo:
    gr.Markdown("## Maximize your auction profits!")

    with gr.Tab("Sale Prediction"):
        gr.Markdown("### Predict when your item will sell")
        realm_id_input = gr.Textbox(label="Realm ID", value=str(config["realm_id"]))
        faction_input = gr.Radio(["Alliance", "Horde"], label="Faction", value="Alliance")
        item_input = gr.Textbox(label="Item ID")
        quantity_input = gr.Number(label="Quantity")
        buyout_input = gr.Number(label="Buyout (in Gold)")
        bid_input = gr.Number(label="Bid (in Gold)")
        time_left_input = gr.Dropdown(time_left_options, label="Time Left")
        predict_button = gr.Button("Predict")
        prediction_output = gr.Textbox(label="AI Prediction")
        predict_button.click(predict_item_sale, inputs=[realm_id_input, faction_input, item_input, quantity_input, buyout_input, bid_input, time_left_input], outputs=prediction_output)
    with gr.Tab("Auction Data"):
        gr.Markdown("### Explore auction data")
        realm_id_input2 = gr.Textbox(label="Realm ID", value=str(config["realm_id"]))
        faction_input2 = gr.Radio(["Alliance", "Horde"], label="Faction", value="Alliance")
        auction_data_button = gr.Button("Get Data")
        auction_data_output = gr.DataFrame(label="Auction Data", interactive=False, height=300)
        auction_data_button.click(lambda realm, faction: auction_data, inputs=[realm_id_input2, faction_input2], outputs=auction_data_output)
    with gr.Tab("Resale Suggestions"):
        gr.Markdown("### Discover resale opportunities")
        realm_id_input3 = gr.Textbox(label="Realm ID", value=str(config["realm_id"]))
        faction_input3 = gr.Radio(["Alliance", "Horde"], label="Faction", value="Alliance")
        desired_profit_input = gr.Number(label="Desired Profit (Gold)")
        suggest_button = gr.Button("Suggest Items")
        suggested_items_output = gr.DataFrame(label="Suggested Items", interactive=False, height=300)
        suggest_button.click(lambda realm, faction, profit: get_suggested_items(realm, faction, profit), inputs=[realm_id_input3, faction_input3, desired_profit_input], outputs=suggested_items_output)
    with gr.Tab("Quick Selling Items"):
        gr.Markdown("### Suggested items for quick sale")
        budget_input = gr.Number(label="Budget (Gold)")
        show_items_button = gr.Button("Show Items")
        quick_selling_items_output = gr.DataFrame(label="Quick Selling Items", interactive=False, height=300)
        show_items_button.click(lambda budget: get_quick_selling_items(budget), inputs=budget_input, outputs=quick_selling_items_output)

demo.launch(share=True)

