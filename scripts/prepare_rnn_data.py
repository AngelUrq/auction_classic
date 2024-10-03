import os, json, torch, time
from datetime import datetime
from tqdm import tqdm

data_dir = "data/auctions"
time_left_mapping = {'VERY_LONG': 48, 'LONG': 12, 'MEDIUM': 2, 'SHORT': 0.5}
auction_appearances = {}
exclude_first_times = [
    '28-01-2024',
    '29-01-2024',
    '30-01-2024',
    '01-03-2024',
    '02-03-2024',
    '23-06-2024',
    '24-06-2024',
    '23-07-2024',
    '24-07-2024',
    '03-08-2024',
    '04-08-2024',
    '11-08-2024',
    '12-08-2024',
    '13-08-2024',
    '15-08-2024',
    '16-08-2024',
    '13-09-2024',
    '14-09-2024',
    '24-09-2024',
    '25-09-2024',
]

def load_auctions(data_dir):
    file_info = {}
    auction_appearances = {}

    for root, dirs, files in os.walk(data_dir):
        for filename in tqdm(files):
            filepath = os.path.join(root, filename)
            date = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')
            file_info[filepath] = date

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}

    print(file_info)
    
    for filepath in tqdm(list(file_info.keys())):
        with open(filepath, 'r') as f:
            try:
                json_data = json.load(f)
                
                if 'auctions' not in json_data:
                    print(f"File {filepath} does not contain 'auctions' key, skipping.")
                    continue
                
                auctions = json_data['auctions']
                timestamp = file_info[filepath]

                for auction in auctions:
                    auction_id = auction['id']
                    auction['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    
                    if auction_id not in auction_appearances:
                        auction_appearances[auction_id] = {
                            'first_datetime': timestamp, 
                            'last_datetime': timestamp
                        }
                    else:
                        auction_appearances[auction_id]['last_datetime'] = timestamp
                
            except json.JSONDecodeError as e:
                print(f"Error loading file {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error loading file {filepath}: {e}")
                continue

    return total_auctions

def process_auction_data(total_auctions):
    auctions_by_item = {}
    hours_on_sale = {}
    auction_ids_by_item = {}
    hours_since_first_appearance_values = [] 

    for auction in total_auctions:

        if not isinstance(auction, dict) or 'item' not in auction or 'id' not in auction['item']:
            print(f"Unexpected structure in auction: {auction}")
            continue

        auction_id = auction['id']
        item_id = auction['item']['id']
        time_left_numeric = time_left_mapping.get(auction['time_left'], 0)
        bid = auction['bid'] * 10000 / 1000
        buyout = auction['buyout'] * 10000 / 1000
        quantity = auction['quantity'] / 200
        time_left = time_left_numeric / 48
        item_index = item_to_index.get(item_id, 1)
        timestamp = datetime.strptime(auction['timestamp'], "%Y-%m-%d %H:%M:%S")

        if timestamp != prediction_time:
            continue

        hours_since_first_appearance = (prediction_time - auction_appearances[auction_id]['first']).total_seconds() / 3600
        hours_since_first_appearance_values.append(hours_since_first_appearance)  
        hours_since_first_appearance_normalized = hours_since_first_appearance / 48.0
        hours_on_sale[auction_id] = (auction_appearances[auction_id]['last'] - auction_appearances[auction_id]['first']).total_seconds() / 3600

        datetime_str = prediction_time.strftime("%Y-%m-%d %H:%M:%S")
        if (item_id, datetime_str) in weekly_historical_prices.index:
            historical_price = weekly_historical_prices.loc[(item_id, datetime_str), 'price']
        else:
            historical_price = buyout

        processed_auction = [
            bid, 
            buyout,  
            quantity, 
            item_index,
            time_left, 
            hours_since_first_appearance_normalized,  
            historical_price  
        ]
        
        if item_index not in auctions_by_item:
            auctions_by_item[item_index] = []
            auction_ids_by_item[item_index] = []

        auctions_by_item[item_index].append(processed_auction)
        auction_ids_by_item[item_index].append(auction_id)


    return auctions_by_item, auction_ids_by_item, hours_on_sale

if __name__ == "__main__":
    start_time = time.time()
    load_auctions()
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
