import argparse
import json
import mysql.connector
import os
import time
from datetime import datetime
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Process auction data.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing the JSON files')
    args = parser.parse_args()
    config = json.load(open("config.json", "r"))  

    db = mysql.connector.connect(**config["database"])

    cursor = db.cursor()
    
    processed_files_path = "processed_files.txt"
    
    if not os.path.exists(processed_files_path):
        with open(processed_files_path, 'w') as file:
            file.write("")
    
    with open(processed_files_path, 'r') as file:
        processed_files = file.readlines()
        
    processed_files = [x.strip() for x in processed_files]

    file_info = {}

    for root, dirs, files in os.walk(args.data_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                date = datetime.strptime(filename[:-5], "%Y%m%dT%H")
                file_info[filepath] = date

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}

    json_files = list(file_info.keys())

    for file in json_files:
        print(file)
        
    processed_auctions = {}
    processed_action_events = {}

    for filepath in tqdm(json_files):
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading file {filepath}: {e}")
            continue

        print(f"Processing file: {filepath}")
        
        if filepath in processed_files:
            print(f"File {filepath} has already been processed. Skipping...")
            continue

        filename = os.path.basename(filepath)
        auction_record = datetime.strptime(filename[:-5], "%Y%m%dT%H")

        auctions_data = []
        action_events_data = []
        
        start_time = time.time()
        for auction in data["auctions"]:
            if processed_auctions.get(auction["id"]) is None:
                processed_auctions[auction["id"]] = True
                auctions_data.append((auction["id"], auction["bid"], auction["buyout"], auction["quantity"], auction["item"]["id"]))
                
            auction_datetime = auction_record.strftime('%Y-%m-%d %H:%M:%S')
            
            if processed_action_events.get(str(auction["id"]) + auction_datetime) is None:
                processed_action_events[str(auction["id"]) + auction_datetime] = True
                action_events_data.append((auction["id"], auction_datetime, auction["time_left"]))
        end_time = time.time()
        
        print(f"Data processed for file {filepath}. Time taken: {end_time - start_time:.2f} seconds. Number of records: {len(data['auctions'])}")

        try:
            start_time = time.time()
            cursor.executemany("""
                INSERT INTO Auctions (auction_id, bid, buyout, quantity, item_id)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    item_id = VALUES(item_id)
            """, auctions_data)
                        
            cursor.executemany("""
                INSERT INTO ActionEvents (auction_id, record, time_left)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    record = VALUES(record)
            """, action_events_data)
            
            db.commit()
            
            end_time = time.time()

            print(f"Auction data for file {filepath} successfully inserted in Auctions. Time taken: {end_time - start_time:.2f} seconds. Number of records: {len(auctions_data)}")
            print(f"Auction events for file {filepath} successfully inserted in ActionEvents. Time taken: {end_time - start_time:.2f} seconds. Number of records: {len(action_events_data)}")
            print("--------------------------------------------------")
            
            with open(processed_files_path, "a") as file:
                file.write(filepath + "\n")
        except mysql.connector.Error as err:
            db.rollback()
            print(f"Error inserting auction data for file {filepath} in Auctions: {err}")

    cursor.close()
    db.close()

if __name__ == "__main__":
    main()
