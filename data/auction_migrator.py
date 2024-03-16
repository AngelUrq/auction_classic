import argparse
import json
import mysql.connector
import os
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

    for filepath in tqdm(json_files):
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading file {filepath}: {e}")
            continue

        print(f"Processing file: {filepath}")

        filename = os.path.basename(filepath)
        auction_record = datetime.strptime(filename[:-5], "%Y%m%dT%H")

        auctions_data = [
            (auction["id"], auction["bid"], auction["buyout"], auction["quantity"], auction["time_left"], auction["item"]["id"])
            for auction in data["auctions"]
        ]
        action_events_data = [(auction["id"], auction_record.strftime('%Y-%m-%d %H:%M:%S')) for auction in data["auctions"]]

        try:
            cursor.executemany("""
                INSERT INTO Auctions (auction_id, bid, buyout, quantity, time_left, item_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    item_id = VALUES(item_id)
            """, auctions_data)
            db.commit()
            print(f"Auction data from file {filepath} successfully inserted into Auctions.")
        except mysql.connector.Error as err:
            db.rollback()
            print(f"Error inserting auction data for file {filepath} in Auctions: {err}")

        try:
            cursor.executemany("""
                INSERT INTO ActionEvents (auction_id, record)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE
                    record = VALUES(record)
            """, action_events_data)
            db.commit()
            print(f"Auction events for file {filepath} successfully inserted in ActionEvents.")
        except mysql.connector.Error as err:
            db.rollback()
            print(f"Error inserting auction events for file {filepath} in ActionEvents: {err}")

    cursor.close()
    db.close()

if __name__ == "__main__":
    main()


