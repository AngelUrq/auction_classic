import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
import random
import os
import warnings
import sys
from pathlib import Path

wd = Path(os.path.dirname(os.path.abspath("__file__"))).parent.resolve()
sys.path.append(str(wd))

from data.transformers import add_features, transform_data

warnings.filterwarnings('ignore')

class AuctionEnv(gym.Env):

    def __init__(self, initial_gold=1000, bankruptcy_penalty=-15000, auction_house="factioned", auction_duration=24, base_path='data/auctions'):
        super(AuctionEnv, self).__init__()

        self.gold = initial_gold
        self.initial_gold = initial_gold 
        self.bankruptcy_penalty = bankruptcy_penalty
        self.current_step = 0
        self.auction_house = auction_house
        self.auction_duration = auction_duration
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.model = pickle.load(open('forest_model.pkl', 'rb'))
        self.items_df = pd.read_csv('data/items.csv')
        self.base_path = base_path
        
        self.lambda_value = 0.0401

    def load_data(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        filename = os.path.basename(filepath)
        auction_record = datetime.strptime(filename[:-5], "%Y%m%dT%H")  

        extracted_data = []
        for item in data["auctions"]:
            extracted_data.append({
                'auction_id': item['id'],
                'item_id': item['item']['id'],
                'bid_in_gold': item['bid'] / 10000.0,
                'buyout_in_gold': item['buyout'] / 10000.0,
                'quantity': item['quantity'],
                'time_left': item['time_left'],
                'first_appearance_timestamp': auction_record,
                'first_appearance_year': auction_record.year,
                'first_appearance_month': auction_record.month,
                'first_appearance_day': auction_record.day,
                'first_appearance_hour': auction_record.hour,
                'listing_duration': 0,
                'unit_price': item['buyout'] / 10000.0 / item['quantity']
            })

        df = pd.DataFrame(extracted_data)
        df['item_id'] = df['item_id'].astype(str)
        self.items_df['item_id'] = self.items_df['item_id'].astype(str)
        df = df.merge(self.items_df, on='item_id', how='left')
        df = df.fillna(0)

        def transform_time_left(df):
            df['time_left'] = np.where(df['time_left'] == 'SHORT', 2, df['time_left'])
            df['time_left'] = np.where(df['time_left'] == 'MEDIUM', 12, df['time_left'])
            df['time_left'] = np.where(df['time_left'] == 'LONG', 24, df['time_left'])
            df['time_left'] = np.where(df['time_left'] == 'VERY_LONG', 48, df['time_left'])

            return df
        
        df = transform_time_left(df)

        return df

    def _get_obs(self):
        if self.current_step >= len(self.auctions_df):
            return np.zeros(self.observation_space.shape)
        
        auction = self.auctions_df.iloc[self.current_step].copy()
    
        required_columns = [
            'item_id', 'buyout_in_gold', 'time_left', 'bid_in_gold', 'quantity', 'unit_price',
            'avg_competitor_price', 'std_competitor_price', 'median_buyout_price', 'median_unit_price', 
            'median_bid_price', 'top_competitor_price', 'lowest_competitor_price', 
            'relative_price_top_competitor', 'relative_price_lowest_competitor', 'rank_bid_price', 
            'rank_buyout_price', 'rank_unit_price', 'competitor_count'
        ]
        
        observation = np.array([self.gold] + [auction.get(column, 0) for column in required_columns])
        return observation

    def _get_info(self):
        return {
            "current_step": self.current_step,
            "gold": self.gold
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gold = self.initial_gold 
        self.current_step = 0
        
        random_folder = random.choice(os.listdir(self.base_path))
        random_json_file = os.path.join(self.base_path, random_folder, random.choice(os.listdir(os.path.join(self.base_path, random_folder))))
        print(f"Selected JSON file: {random_json_file}")

        self.auctions_df = self.load_data(random_json_file)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def calculate_deposit(self, msv):
        if self.auction_house == "factioned":
            if self.auction_duration == 12:
                return 0.15 * msv
            elif self.auction_duration == 24:
                return 0.30 * msv
            elif self.auction_duration == 48:
                return 0.60 * msv
        elif self.auction_house == "neutral":
            if self.auction_duration == 12:
                return 0.75 * msv
            elif self.auction_duration == 24:
                return 1.50 * msv
            elif self.auction_duration == 48:
                return 3.00 * msv
        return 0

    def calculate_auction_house_cut(self, amount):
        if self.auction_house == "factioned":
            return 0.05 * amount
        elif self.auction_house == "neutral":
            return 0.15 * amount
        return 0

    def step(self, action):
        if self.current_step >= len(self.auctions_df):
            done = True
            reward = 0
            print(f"Step {self.current_step}: No more auctions left. Ending episode.")
            return self._get_obs(), reward, done, False, self._get_info()

        auction = self.auctions_df.iloc[self.current_step].copy()
        buyout_price = auction['buyout_in_gold']
        quantity = auction['quantity']
        msv = quantity * (auction.get('sell_price_gold', 0) + auction.get('sell_price_silver', 0) / 100)
        truncated = False
        reward = 0

        deposit = self.calculate_deposit(msv)

        total_cost = buyout_price + deposit
        
        print(f"Current gold: {self.gold}")
        print(f"Buyout price: {buyout_price}, Quantity: {quantity}, Total cost: {total_cost}")
        print(f"Action taken (bid price): {action[0]}")

        if action[0] > 1 and self.gold >= total_cost:
            self.gold -= total_cost

            new_item_data = auction.copy()
            new_item_data['buyout_in_gold'] = action[0]
            new_item_data['bid_in_gold'] = action[0]
            new_item_data['unit_price'] = action[0] / quantity

            temp_auctions_df = self.auctions_df.copy()
            temp_auctions_df.loc[self.current_step] = new_item_data
            temp_auctions_df = add_features(temp_auctions_df)
            
            categorical_columns = ['quality', 'item_class', 'item_subclass', 'is_stackable']
            temp_auctions_df[categorical_columns] = temp_auctions_df[categorical_columns].astype(str)
    
            X, _ = transform_data(temp_auctions_df)
            X = X[-1].reshape(1, -1)
            predicted_hours = self.model.predict(X)[0]
            sale_probability = np.exp(-self.lambda_value * predicted_hours)
            
            print(f"Sale probability: {sale_probability:.2f}")
            sold = random.random() < sale_probability

            if sold:
                auction_house_cut = self.calculate_auction_house_cut(action[0])
                reward = action[0] - buyout_price - auction_house_cut
                
                self.gold += reward + deposit  
                print(f"Item sold. Profit: {reward}, Auction house cut: {auction_house_cut}, Remaining gold: {self.gold}")
            else:
                reward = -total_cost
                print(f"Item not sold. Loss: {reward}, Remaining gold: {self.gold}")

        self.current_step += 1
        done = self.current_step >= len(self.auctions_df)
        observation = self._get_obs()
        info = self._get_info()

        if self.gold <= 0:
            reward += self.bankruptcy_penalty
            truncated = True
            done = True
            print(f"Item not sold. Loss: {reward}, Remaining gold: {self.gold}")

        return observation, reward, done, truncated, info

    def render(self):
        print(f'Step: {self.current_step}, Gold: {self.gold}')

    def close(self):
        pass

from gymnasium.envs.registration import register

register(
    id='AuctionEnv-v0',
    entry_point='__main__:AuctionEnv'
)

if __name__ == "__main__":
    env = gym.make('AuctionEnv-v0')

    state, info = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

    env.close()
    print(f'Total reward: {total_reward:.2f}')
