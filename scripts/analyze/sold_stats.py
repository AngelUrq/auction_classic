import json
import argparse
from datetime import datetime
from tqdm import tqdm

TIME_LEFT_TO_INT = {
    'VERY_LONG': 48.0,
    'LONG': 12.0,
    'MEDIUM': 2.0,
    'SHORT': 0.5,
}
EXPIRED_LISTING_DURATIONS = [11.0, 23.0, 47.0]

def is_expired(listing_duration, time_left):
    return 1.0 if (float(listing_duration) in EXPIRED_LISTING_DURATIONS and float(time_left) <= 0.5) else 0.0

def main():
    parser = argparse.ArgumentParser(description="Analyze is_sold statistics")
    parser.add_argument("--timestamps", type=str, default="generated/timestamps.json")
    args = parser.parse_args()

    print(f"Loading {args.timestamps}...")
    try:
        with open(args.timestamps, 'r') as f:
            timestamps = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.timestamps} not found.")
        return

    total_auctions = len(timestamps)
    total_sold = 0
    total_expired = 0
    total_canceled = 0

    print(f"Analyzing {total_auctions:,} auctions...")
    for auction_id, data in tqdm(timestamps.items(), total=total_auctions, desc="Calculating stats"):
        first_app = datetime.strptime(data['first_appearance'], '%Y-%m-%d %H:%M:%S')
        last_app = datetime.strptime(data['last_appearance'], '%Y-%m-%d %H:%M:%S')
        listing_duration = min(int((last_app - first_app).total_seconds() / 3600), 47)
        
        last_time_left_str = data.get('last_time_left', 'SHORT')
        final_time_left_val = float(TIME_LEFT_TO_INT.get(last_time_left_str, 0.5))
        
        is_expired_val = is_expired(listing_duration, final_time_left_val)
        final_buyout_rank = float(data.get('last_buyout_rank', 1.0))
        
        # Exclude MEDIUMs by using > 2.0
        if is_expired_val == 0.0 and final_buyout_rank == 0.0 and final_time_left_val > 2.0:
            is_sold_val = 1.0
        else:
            is_sold_val = 0.0
            
        if is_sold_val == 1.0:
            total_sold += 1
        elif is_expired_val == 1.0:
            total_expired += 1
        else:
            total_canceled += 1

    print("\n" + "-" * 40)
    print(f"Total Auctions: {total_auctions:,}")
    if total_auctions > 0:
        print(f"Sold (Proxy Label): {total_sold:,} ({total_sold/total_auctions*100:.2f}%)")
        print(f"Expired:            {total_expired:,} ({total_expired/total_auctions*100:.2f}%)")
        print(f"Canceled (or other):{total_canceled:,} ({total_canceled/total_auctions*100:.2f}%)")
    print("-" * 40)

if __name__ == '__main__':
    main()
