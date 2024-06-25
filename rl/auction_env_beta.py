import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import random
from datetime import datetime
import pandas as pd
import pickle

LAMBDA = 0.0401
items_df = pd.read_csv('data/items.csv')
model = pickle.load(open('forest_model.pkl', 'rb'))

def load_and_prepare_data_from_json(item_data=None):
    try:
        with open(os.path.join(os.path.dirname(__file__), '20231104T13.json'), 'r') as file:
            auction_data = json.load(file).get('auctions', [])
        
        if not auction_data:
            raise ValueError("No auction data found in the JSON file.")
        df = pd.DataFrame(auction_data)
        if item_data:
            df = pd.concat([df, pd.DataFrame([item_data])], ignore_index=True)
        
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['first_appearance_timestamp'] = current_timestamp
        df['first_appearance_year'] = int(current_timestamp[:4])
        df['first_appearance_month'] = int(current_timestamp[5:7])
        df['first_appearance_day'] = int(current_timestamp[8:10])
        df['first_appearance_hour'] = int(current_timestamp[11:13])
        df['hours_on_sale'] = 0
        df['unit_price'] = df['buyout_in_gold'] / df['quantity']
        df['unit_price'].fillna(0, inplace=True)
        df = df.merge(items_df, on='item_id', how='inner')
        df = add_features(df)
        X, _ = transform_data(df)

        df['predicted_hours'] = model.predict(X)
        return df
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading and preparing data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def convert_prediction_to_probability(predicted_hours):
    return np.exp(-LAMBDA * predicted_hours)

def predict_item_sale(item, quantity, buyout, bid, time_left_hours, num_simulations=100):
    try:
        item_id = int(item)
        if item_id not in items_df['item_id'].values:
            return "Error: The item is not found in the database. Please check the data."

        item_data = {
            'auction_id': 0,
            'item_id': item_id,
            'quantity': quantity,
            'buyout_in_gold': buyout,
            'bid_in_gold': bid,
            'time_left': time_left_hours
        }

        probabilities = []
        for _ in range(num_simulations):
            df = load_and_prepare_data_from_json(item_data)
            if df is not None:
                predicted_hours = df['predicted_hours'].iloc[-1]
                probability = convert_prediction_to_probability(predicted_hours)
                probabilities.append(probability)

        average_probability = np.mean(probabilities)
        return f"The average sale probability of the item within the allowed time is {average_probability * 100:.2f}%."
    except ValueError:
        return "Error: The item ID must be an integer."
    except Exception as e:
        return f"Error calculating sale probability: {e}"

if "AuctionEnv-v0" not in gym.envs.registry.keys():
    gym.register(id="AuctionEnv-v0", entry_point="__main__:AuctionEnv")

class AuctionEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, json_file="20231104T13.json", starting_gold=1000, auction_house_type="Faction"):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 48, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        )
        try:
            with open(os.path.join(os.path.dirname(__file__), json_file), 'r') as file:
                data = json.load(file)
                self.auction_data = data.get("auctions", [])
                if not self.auction_data:
                    raise ValueError("The JSON file does not contain auction data.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading auction data: {e}")
            self.auction_data = []
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.auction_data = []

        self._validate_auction_data()
        self.starting_gold = starting_gold
        self.auction_house_type = auction_house_type
        self.auction_fee = 0.1
        self.deposit_fees = {
            "Faction": {12: 0.15, 24: 0.30, 48: 0.60},
            "Neutral": {12: 0.75, 24: 1.50, 48: 3.00}
        }
        self.time_left_mapping = {'SHORT': 12, 'MEDIUM': 24, 'LONG': 48, 'VERY_LONG': 48}
        self.reset()
    def _validate_auction_data(self):
        required_keys = ["id", "item", "bid", "buyout", "quantity", "time_left"]
        for i, auction in enumerate(self.auction_data):
            if not isinstance(auction, dict):
                raise ValueError(f"The auction at index {i} is not a valid dictionary.")
            for key in required_keys:
                if key not in auction:
                    raise ValueError(f"The auction at index {i} does not have the key '{key}'.")
                if key == "item" and ("id" not in auction["item"] or not isinstance(auction["item"], dict)):
                    raise ValueError(f"The item in the auction {i} does not have a valid ID.")

    def _calculate_reward(self, action, state):
        buyout_price = state[2]
        time_left = state[3]
        merchant_sell_value = state[7]
        item_cost = state[6] * state[5]

        if action[0] <= 0:
            return 0
        if action[0] < buyout_price:
            return -10

        selling_price = action[0]
        probability_of_selling = max(0, 1 - (selling_price / (buyout_price + 1e-6)))
        sold = random.random() < probability_of_selling

        deposit_fee_percentage = self.deposit_fees.get(self.auction_house_type, {}).get(time_left, 0)
        deposit_fee = merchant_sell_value * deposit_fee_percentage

        if self.gold < deposit_fee:
            return -20
        self.gold -= deposit_fee
        auction_fee = self.auction_fee * selling_price if sold else 0
        profit = selling_price - auction_fee - item_cost if sold else -deposit_fee - item_cost
        reward = profit
        if profit < -50:
            reward -= 50
        self.episode_profits.append(profit)
        return reward

    def _get_auction_state(self, auction_index):
        auction_info = self.auction_data[auction_index]

        item_id = auction_info.get('item', {}).get('id', 0)
        buyout_price = auction_info.get('buyout', 0)
        time_left = self.time_left_mapping.get(auction_info.get('time_left', 'SHORT'), 12)
        bid = auction_info.get('bid', 0)
        quantity = auction_info.get('quantity', 1)
        unit_price = buyout_price / quantity if quantity > 0 else 0
        item_row = items_df[items_df['item_id'] == item_id]
        merchant_sell_value = (item_row['sell_price_gold'].values[0] + item_row['sell_price_silver'].values[0] / 100) if not item_row.empty else 0
        state = np.array([
            self.gold, item_id, buyout_price, time_left, bid, quantity, unit_price,
            merchant_sell_value,
        ], dtype=np.float32)

        if not self.observation_space.contains(state):
            raise ValueError(f"The observation {state} is out of the observation space.")
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gold = self.starting_gold
        self.episode_profits = []
        self.current_auction = random.randrange(len(self.auction_data))
        observation = self._get_auction_state(self.current_auction)
        
        if not self.observation_space.contains(observation):
            raise ValueError(f"The observation {observation} is out of the observation space.")
        return observation, {}

    def step(self, action):
        auction_data = self._get_auction_state(self.current_auction)
        reward = self._calculate_reward(action, auction_data)
        self.current_auction += 1

        terminated = self.gold <= 0 or self.current_auction >= len(self.auction_data)
        truncated = False
        info = {'gold': self.gold}
        observation = self._get_auction_state(self.current_auction) if not terminated else None
        return observation, reward, terminated, truncated, info
    
if __name__ == "__main__":
    env = gym.make("AuctionEnv-v0")
    observation, _ = env.reset()

    for step in range(len(env.auction_data)):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}:")
        print(f"  Action (bid price): {action[0]:.2f}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Current gold: {info['gold']:.2f}")

        if terminated:
            break
    total_profit = sum(env.episode_profits)
    print(f"\nEpisode finished. Total profit: {total_profit:.2f}, Final gold: {info['gold']:.2f}\n")

    env.close()
