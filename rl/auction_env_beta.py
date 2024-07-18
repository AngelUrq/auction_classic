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
from transformers import add_features, transform_data

warnings.filterwarnings('ignore')

class AuctionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, initial_gold=1000, penalizacion_quiebra=-15000, render_mode=None, auction_house="factioned", auction_duration=24):
        super(AuctionEnv, self).__init__()

        self.gold = initial_gold
        self.penalizacion_quiebra = penalizacion_quiebra
        self.current_step = 0
        self.auction_house = auction_house
        self.auction_duration = auction_duration

        self.action_space = spaces.Box(low=0, high=1000000, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(20,), dtype=np.float32)
        
        self.model = pickle.load(open('forest_model.pkl', 'rb'))
        self.items_df = pd.read_csv('data/items.csv')
        
        self.lambda_value = 0.0401
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.reset()

    def load_data(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        extracted_data = []
        for item in data["auctions"]:
            extracted_data.append({
                'auction_id': item['id'],
                'item_id': item['item']['id'],
                'bid_in_gold': item['bid'],
                'buyout_in_gold': item['buyout'],
                'quantity': item['quantity'],
                'time_left': item['time_left']
            })

        df = pd.DataFrame(extracted_data)
        auction_record = datetime.strptime(filename.split(os.sep)[-2], "%d-%m-%Y")
        df['first_appearance_timestamp'] = auction_record
        df['first_appearance_year'] = auction_record.year
        df['first_appearance_month'] = auction_record.month
        df['first_appearance_day'] = auction_record.day
        df['first_appearance_hour'] = auction_record.hour
        df['hours_on_sale'] = 0
        df['unit_price'] = df['buyout_in_gold'] / df['quantity']
        df['item_id'] = df['item_id'].astype(str)
        self.items_df['item_id'] = self.items_df['item_id'].astype(str)
        df = df.merge(self.items_df, on='item_id', how='left')
        df = df.fillna(0)
        return df

    def _get_obs(self):
        if self.current_step >= len(self.auctions_df):
            return np.zeros(self.observation_space.shape)
        auction = self.auctions_df.iloc[self.current_step]
        return auction.to_numpy()

    def _get_info(self):
        return {
            "current_step": self.current_step,
            "gold": self.gold
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gold = 1000
        self.current_step = 0
        
        base_path = 'auctions'
        random_folder = random.choice(os.listdir(base_path))
        random_json_file = os.path.join(base_path, random_folder, random.choice(os.listdir(os.path.join(base_path, random_folder))))
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

    def step(self, action):
        if self.current_step >= len(self.auctions_df):
            done = True
            reward = -100
            print(f"Step {self.current_step}: No more auctions left. Ending episode.")
            return self._get_obs(), reward, done, False, self._get_info()

        auction = self.auctions_df.iloc[self.current_step]
        buyout_price = auction['buyout_in_gold']
        quantity = auction['quantity']
        msv = auction['sell_price_gold']

        deposit = self.calculate_deposit(msv)
        print(f"Calculated deposit: {deposit}")

        total_cost = buyout_price + deposit
        if self.gold < total_cost:
            reward = -1
            self.current_step += 1
            print(f"Step {self.current_step}: Not enough gold to cover the deposit and total cost. Gold: {self.gold}, Total cost: {total_cost}")
            return self._get_obs(), reward, False, False, self._get_info()

        self.gold -= deposit
        print(f"Deposit paid: {deposit}, Remaining gold after paying deposit: {self.gold}")

        print(f"Current gold: {self.gold}")
        print(f"Buyout price: {buyout_price}, Quantity: {quantity}, Total cost: {total_cost}")
        print(f"Action taken (bid price): {action[0]}")

        if action[0] > 0 and self.gold >= total_cost - deposit:
            new_item_data = auction.copy()
            new_item_data['buyout_in_gold'] = action[0]
            new_item_data['unit_price'] = action[0] / quantity

            temp_auctions_df = pd.concat([self.auctions_df, pd.DataFrame([new_item_data])], ignore_index=True)

            temp_auctions_df = add_features(temp_auctions_df)
            
            categorical_columns = ['quality', 'item_class', 'item_subclass', 'is_stackable']
            temp_auctions_df[categorical_columns] = temp_auctions_df[categorical_columns].astype(str)
            
            X, _ = transform_data(temp_auctions_df)
            X_new = X[-1].reshape(1, -1) 
            if X_new.shape[1] != self.model.n_features_in_:
                raise ValueError(f"The model expects {self.model.n_features_in_} features, but got {X_new.shape[1]}")

            predicted_hours = self.model.predict(X_new)[0]
            sale_probability = np.exp(-self.lambda_value * predicted_hours)
            print(f"Sale probability: {sale_probability:.2f}")
            sold = random.random() < sale_probability

            if sold:
                reward = action[0] - buyout_price
                self.gold += reward + deposit  
                print(f"Item sold. Profit: {reward}, Remaining gold: {self.gold}")
            else:
                reward = -buyout_price
                self.gold -= buyout_price
                if self.gold <= 0: 
                    self.gold = 0  
                    reward += self.penalizacion_quiebra  
                print(f"Item not sold. Loss: {reward}, Remaining gold: {self.gold}")
            
            self.current_step += 1
            done = self.current_step >= len(self.auctions_df)
        else:
            reward = -1
            print(f"Step {self.current_step}: Not enough gold to buy. Action: {action[0]}, Total cost: {total_cost}, Gold: {self.gold}")
            self.current_step += 1
            done = self.current_step >= len(self.auctions_df)

        observation = self._get_obs()
        info = self._get_info()

        print(f"State after step {self.current_step}: Gold: {self.gold}, Reward: {reward}")
        return observation, reward, done, False, info

    def render(self):
        print(f'Step: {self.current_step}, Gold: {self.gold}')

    def close(self):
        pass
from gymnasium.envs.registration import register

register(
    id='AuctionEnv-v0',
    entry_point='__main__:AuctionEnv',
    max_episode_steps=1000,
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
