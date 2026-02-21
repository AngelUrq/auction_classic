import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

from datetime import datetime, timedelta
from tqdm import tqdm

def compute_auction_hours(timestamps):
    print('Computing auction hours...')
    data = []

    for auction_id, auction in tqdm(timestamps.items()):
        first_appearance = datetime.strptime(auction['first_appearance'], '%Y-%m-%d %H:%M:%S')
        last_appearance = datetime.strptime(auction['last_appearance'], '%Y-%m-%d %H:%M:%S')

        listing_duration = (last_appearance - first_appearance).total_seconds() / 3600

        data.append((
            auction_id,
            auction['first_appearance'],
            auction['last_appearance'],
            auction['item_id'],
            listing_duration
        ))

    df = pd.DataFrame(data, columns=['auction_id', 'first_appearance', 'last_appearance', 'item_id', 'listing_duration'])

    df['first_appearance'] = pd.to_datetime(df['first_appearance'])
    df['last_appearance'] = pd.to_datetime(df['last_appearance'])

    df = df[df['listing_duration'] <= 50]

    return df

def calculate_weekly_averages(df):
    print('Calculating weekly averages...')
    df['first_appearance'] = pd.to_datetime(df['first_appearance'])

    min_date = df['first_appearance'].min() + pd.Timedelta(days=9)
    max_date = df['first_appearance'].max()

    date_range = pd.date_range(min_date, max_date, freq='D')
    weekly_averages = []

    for current_date in tqdm(date_range):
        end_date = current_date - pd.Timedelta(days=2)
        start_date = current_date - pd.Timedelta(days=9)

        mask = (
            (df['first_appearance'] >= start_date) &
            (df['first_appearance'] <= end_date)
        )
        week_data = df[mask]

        item_averages = week_data.groupby('item_id')['listing_duration'].mean().reset_index()
        item_averages['date'] = current_date

        if len(item_averages) > 0:
            weekly_averages.append(item_averages)

    result = pd.concat(weekly_averages, ignore_index=True)
    return result[['date', 'item_id', 'listing_duration']]

def main(args):
    print('Loading timestamps...')
    with open(args.timestamps, 'r') as f:
        timestamps = json.load(f)

    print('Loading timestamps... Done')

    df = compute_auction_hours(timestamps)
    weekly_averages = calculate_weekly_averages(df)

    print(f'Saving weekly hours to {args.output_dir}/weekly_hours.csv')
    weekly_averages.to_csv(f'{args.output_dir}/weekly_hours.csv', index=False)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weekly hours')
    parser.add_argument('--timestamps', type=str, required=True, help='Path to the timestamps file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output folder')
    args = parser.parse_args()

    main(args)
