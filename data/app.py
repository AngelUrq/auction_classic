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
    df = add_features(df)
    X, _ = transform_data(df)
    user_prediction = model.predict(X)[-1]  
    if user_prediction < 8:
        return f"Your item will sell in approximately {user_prediction:.2f} hours."
    else:
        return f"Your item may take more than {user_prediction:.2f} hours to sell."

def get_suggested_items(realm_id, faction, desired_profit, min_sale_probability, price_filter_min, price_filter_max, item_class=None, item_subclass=None):
    if auction_data is not None:
        lambda_value = 0.0401
        resale_percentage = 1.30
        results = []

        filtered_df = auction_data.copy()

        if item_class:
            filtered_df = filtered_df[filtered_df['item_class'].str.lower() == item_class.lower()]
        if item_subclass:
            filtered_df = filtered_df[filtered_df['item_subclass'].str.lower() == item_subclass.lower()]
        filtered_df = filtered_df[(filtered_df['buyout_in_gold'] >= price_filter_min) & (filtered_df['buyout_in_gold'] <= price_filter_max)]

      
        print("Filtered DataFrame by class, subclass, and price range:")
        print(filtered_df.head())
        print(f"Number of items after filtering: {filtered_df.shape[0]}")

        filtered_df = filtered_df.reset_index(drop=True)

        for index, row in filtered_df.iterrows():
            temp_df = filtered_df.copy()
            temp_df.at[index, 'time_left'] = 24
            new_buyout_price = row['buyout_in_gold'] * resale_percentage
            temp_df.at[index, 'buyout_in_gold'] = new_buyout_price

            print(f"DataFrame before add_features at index {index}:")
            print(temp_df.head())
            temp_df = add_features(temp_df)
            print(f"DataFrame after add_features at index {index}:")
            print(temp_df.head())

            X, _ = transform_data(temp_df)
            if X.shape[0] == 0:
                print("No data in transformed X.")
                continue
            if index >= X.shape[0]:
                print(f"Index {index} out of bounds for transformed data.")
                continue
            X_item = X[index]
            prediction = model.predict([X_item])
            probability_of_sale = np.exp(-lambda_value * prediction[0])


            print(f"Probability of sale for item_id {row['item_id']} at index {index}: {probability_of_sale}")
            if probability_of_sale < min_sale_probability:
                print(f"Item excluded due to low probability of sale: {probability_of_sale}")
                continue
            profit = new_buyout_price - row['buyout_in_gold']
            print(f"Profit for item_id {row['item_id']} at index {index}: {profit}")

            if profit < desired_profit:
                print(f"Item excluded due to low profit: {profit}")
                continue
            results.append({
                'item_id': row['item_id'],
                'item_name': row['item_name'],
                'quantity': row['quantity'],
                'original_buyout_price': row['buyout_in_gold'],
                'new_buyout_price': new_buyout_price,
                'probability_of_sale': probability_of_sale,
                'prediction': prediction[0]
            })


        print("Results before sorting:")
        print(results)
        results = sorted(results, key=lambda x: x['probability_of_sale'], reverse=True)
        print("Results after sorting:")
        print(results)

        return pd.DataFrame(results)
    else:
        print("Auction data not available.")
        return pd.DataFrame()

def get_quick_selling_items(budget):
    df = load_and_prepare_data()
    if df is not None:
        df = add_features(df)
        X, _ = transform_data(df)
        df['prediction'] = model.predict(X)

        quick_selling_items = df[(df['prediction'] < 8) & (df['buyout_in_gold'] <= budget)]
        print("Quick Selling Items DataFrame:")
        print(quick_selling_items.head()) 
        return quick_selling_items[['item_id', 'item_name', 'quantity', 'buyout_in_gold', 'bid_in_gold', 'time_left', 'prediction']]
    else:
        print("Error: Could not load auction data.")
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
        category_input = gr.Dropdown(choices=items_df['item_class'].unique().tolist(), label="Item Class", multiselect=False)
        subclass_input = gr.Dropdown(choices=items_df['item_subclass'].unique().tolist(), label="Item Subclass", multiselect=False)
        desired_profit_input = gr.Number(label="Desired Profit (Gold)")
        sale_probability_input = gr.Slider(label="Minimum Sale Probability (%)", minimum=0, maximum=100, step=1, value=80)
        price_filter_min_input = gr.Number(label="Minimum Original Buyout Price (Gold)", value=0)
        price_filter_max_input = gr.Number(label="Maximum Original Buyout Price (Gold)", value=10000)
        suggest_button = gr.Button("Suggest Items")
        suggested_items_output = gr.DataFrame(label="Suggested Items", interactive=False, height=300)
        suggest_button.click(lambda realm, faction, category, subclass, profit, prob, min_price, max_price: get_suggested_items(realm, faction, profit, prob / 100, min_price, max_price, category, subclass), inputs=[realm_id_input3, faction_input3, category_input, subclass_input, desired_profit_input, sale_probability_input, price_filter_min_input, price_filter_max_input], outputs=suggested_items_output)
    with gr.Tab("Quick Selling Items"):
        gr.Markdown("### Suggested items for quick sale")
        budget_input = gr.Number(label="Budget (Gold)")
        show_items_button = gr.Button("Show Items")
        quick_selling_items_output = gr.DataFrame(label="Quick Selling Items", interactive=False, height=300)
        show_items_button.click(lambda budget: get_quick_selling_items(budget), inputs=budget_input, outputs=quick_selling_items_output)

demo.launch(share=True)
