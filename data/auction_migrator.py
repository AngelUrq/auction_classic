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
        print(f"Processing file: {file}")
        
        try:
            with open(file, "r") as json_file:
                data = json.load(json_file)
                print(f"Loaded JSON data from {file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading file {file}: {e}")
            with open("error_log.txt", "a") as log_file:
                log_file.write(f"Error reading file {file}: {e}\n")
            continue

        filename = os.path.basename(file)
        auction_record = datetime.strptime(filename[:-5], "%Y%m%dT%H")
        print(f"Parsed auction record datetime: {auction_record}")
        processed_auctions = {}
        processed_action_events = {}
        auctions_data = []
        action_events_data = []
        
        start_time = time.time()
        for auction in data.get("auctions", []):
            if processed_auctions.get(auction["id"]) is None:
                processed_auctions[auction["id"]] = True
                auctions_data.append((auction["id"], auction["bid"], auction["buyout"], auction["quantity"], auction["item"]["id"]))
                print(f"Added auction {auction['id']} to auctions_data")
                
            auction_datetime = auction_record.strftime('%Y-%m-%d %H:%M:%S')
            
            if processed_action_events.get(str(auction["id"]) + auction_datetime) is None:
                processed_action_events[str(auction["id"]) + auction_datetime] = True
                action_events_data.append((auction["id"], auction_datetime, auction["time_left"]))
                print(f"Added auction event {auction['id']} at {auction_datetime} to action_events_data")
        
        end_time = time.time()
        print(f"Data processed for file {file}. Time taken: {end_time - start_time:.2f} seconds. Number of records: {len(data['auctions'])}")

        try:
            start_time = time.time()
            cursor.executemany("""
                INSERT INTO Auctions (auction_id, bid, buyout, quantity, item_id)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    item_id = VALUES(item_id)
            """, auctions_data)
            print(f"Inserted {len(auctions_data)} records into Auctions")
                        
            cursor.executemany("""
                INSERT INTO ActionEvents (auction_id, record, time_left)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    record = VALUES(record)
            """, action_events_data)
            print(f"Inserted {len(action_events_data)} records into ActionEvents")
            
            db.commit()
            print(f"Committed transactions for file {file}")
            end_time = time.time()
            print(f"Auction data for file {file} successfully inserted in Auctions. Time taken: {end_time - start_time:.2f} seconds. Number of records: {len(auctions_data)}")
            print(f"Auction events for file {file} successfully inserted in ActionEvents. Time taken: {end_time - start_time:.2f} seconds. Number of records: {len(action_events_data)}")
            print("--------------------------------------------------")
            
        except mysql.connector.Error as err:
            db.rollback()
            print(f"Error inserting auction data for file {file} in Auctions: {err}")
            with open("error_log.txt", "a") as log_file:
                log_file.write(f"Error in {file}\n")
                log_file.write(f"Error: {err}\n")
                log_file.write("Auctions data:\n")
                log_file.write(json.dumps(auctions_data, indent=4) + "\n")
                log_file.write("ActionEvents data:\n")
                log_file.write(json.dumps(action_events_data, indent=4) + "\n")

    cursor.close()
    db.close()

if __name__ == "__main__":
    main()
