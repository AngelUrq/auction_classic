import argparse
import json
import mysql.connector
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Process auction data.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing the JSON files')
    args = parser.parse_args()
    config = json.load(open("config.json", "r"))#database
    
    #Connect to the database using values from config.json
    db = mysql.connector.connect(**config["database"])

    cursor = db.cursor()

    #Iterate through all JSON files in the data directory and its subdirectories
    for root, dirs, files in os.walk(args.data_dir):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                try:
                    data = json.load(open(filepath, "r"))
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading file {filepath}: {e}")
                    continue

                print(f"Processing file: {filepath}")

                #Extract the auction timestamp from the filename
                auction_record = datetime.strptime(filename[:-5], "%Y%m%dT%H")

                #Prepare auction data for insertion
                auctions_data = [(auction["id"], auction["bid"], auction["buyout"], auction["quantity"], auction["time_left"]) for auction in data["auctions"]]
                action_events_data = [(auction["id"], auction_record.strftime('%Y-%m-%d %H:%M:%S')) for auction in data["auctions"]]

                #Insert auction data into the Auctions table
                try:
                    cursor.executemany("""
                        INSERT INTO Auctions (auction_id, bid, buyout, quantity, time_left)
                        VALUES (%s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            bid = VALUES(bid),
                            buyout = VALUES(buyout),
                            quantity = VALUES(quantity),
                            time_left = VALUES(time_left)
                    """, auctions_data)
                    db.commit()
                    print(f"Auction data from file {filepath} successfully inserted into Auctions.")
                except mysql.connector.Error as err:
                    db.rollback()
                    print(f"Error inserting auction data for file {filepath} in Auctions: {err}")

                #Insert data into ActionEvents
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

    #Close the cursor and the database connection
    cursor.close()
    db.close()

if __name__ == "__main__":#add main 
    main()
