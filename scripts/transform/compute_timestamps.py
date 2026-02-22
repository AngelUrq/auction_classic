import os, json, time
import argparse
from datetime import datetime
from tqdm import tqdm

def process_auctions(args):
    file_info = {}
    auction_appearances = {}
    data_dir = args.data_dir

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            date = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')
            file_info[filepath] = date

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}

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
                            'first_appearance': timestamp.strftime("%Y-%m-%d %H:%M:%S"), 
                            'last_appearance': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            'last_time_left': auction['time_left'],
                            'item_id': auction['item']['id'],
                        }
                    else:
                        auction_appearances[auction_id]['last_appearance'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        auction_appearances[auction_id]['last_time_left'] = auction['time_left']
                
            except json.JSONDecodeError as e:
                print(f"Error loading file {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error loading file {filepath}: {e}")
                continue

    return auction_appearances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check JSON files in auctions folder')
    parser.add_argument('--data_dir', type=str, default='data/auctions/', help='Path to the auctions folder')
    parser.add_argument('--output_file', type=str, default='generated/timestamps.json', help='Path to the output file')
    args = parser.parse_args()

    if os.path.exists(args.output_file):
        print(f"Skipping: {args.output_file} already exists.")
        exit(0)

    start_time = time.time()
    auction_appearances = process_auctions(args)

    with open(args.output_file, 'w') as f:
        json.dump(auction_appearances, f, indent=4)

    print(f"Execution time: {time.time() - start_time:.2f} seconds")
