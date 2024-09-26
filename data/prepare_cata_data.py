import os, json, torch, time
from datetime import datetime

BASE_DIR = "/home/cmiranda/Documents/auctions"
TIME_LEFT_MAPPING = {'VERY_LONG': 48, 'LONG': 12, 'MEDIUM': 2, 'SHORT': 0.5}
auction_appearances = {}

def process_json_file(file_path, file_timestamp):
    try:
        with open(file_path, 'r') as f:
            auctions = json.load(f).get('auctions', [])
        return {auction['item']['id']: [
            auction['bid'], auction['buyout'], auction['quantity'], auction['item']['id'],
            TIME_LEFT_MAPPING.get(auction['time_left'], 0),
            (file_timestamp - auction_appearances.setdefault(auction['id'], {'first': file_timestamp, 'last': file_timestamp})['first']).total_seconds() / 3600,
            (auction_appearances[auction['id']].update({'last': file_timestamp}) or auction_appearances[auction['id']]['last'] - auction_appearances[auction['id']]['first']).total_seconds() / 3600
        ] for auction in auctions}
    except json.JSONDecodeError as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

def prepare_data(base_dir):
    sequences_dir = os.path.join(base_dir, "sequences")
    os.makedirs(sequences_dir, exist_ok=True)
    file_count, processed_count = 0, 0
    
    for date_folder in os.listdir(base_dir):
        date_folder_path = os.path.join(base_dir, date_folder)
        if not os.path.isdir(date_folder_path):
            continue
        
        output_folder = os.path.join(sequences_dir, date_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        for file_name in os.listdir(date_folder_path):
            if not file_name.endswith('.json'):
                continue
            
            file_count += 1
            file_path = os.path.join(date_folder_path, file_name)
            file_timestamp = datetime.strptime(f"{file_name.split('T')[0]}T{file_name.split('T')[1].split('.')[0]}", "%Y%m%dT%H")
            file_start_time = time.time()
            auctions_by_item = process_json_file(file_path, file_timestamp)
            print(f"Processed {file_name} in {time.time() - file_start_time:.2f} seconds")
            
            if auctions_by_item:
                processed_count += 1
                torch.save(auctions_by_item, os.path.join(output_folder, f"{file_name.split('T')[1].split('.')[0]}.pt"))
        
        print(f"Processed folder: {date_folder}")
    
    print(f"Total files: {file_count}, Processed files: {processed_count}")

if __name__ == "__main__":
    start_time = time.time()
    prepare_data(BASE_DIR)
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
