import os
import json
import mysql.connector
import logging

# Configure logs
logging.basicConfig(filename='migration_log.txt', level=logging.INFO)

def migrate_data(json_directory, table_name, conn):
    cursor = conn.cursor()

    # Log start
    logging.info(f"Starting data migration for table {table_name}...")

    try:
        # Iterate over all JSON files in the directory
        for filename in os.listdir(json_directory):
            if filename.endswith(".json"):
                logging.info(f"Processing file: {filename}")
                with open(os.path.join(json_directory, filename), 'r') as file:
                    try:
                        # Load JSON data from the file
                        data = json.load(file)
                        logging.info(f"Data loaded from {filename}: {data}")

                        # Access auction data nested under the 'item' key
                        item_data = data.get('item', {})

                        # Use the unique ID from the file as the auction identifier
                        auction_id = data.get('id')

                        if auction_id is not None:
                            # Check if the entry already exists in the database
                            query_check_duplicate = f"SELECT 1 FROM {table_name} WHERE auction_id = %s"
                            cursor.execute(query_check_duplicate, (auction_id,))
                            result = cursor.fetchone()

                            if result:
                                # Duplicate entry found, consider updating.
                                logging.warning(f"Duplicate entry found for {filename}. Consider updating.")
                            else:
                                # Build insertion query with parameters
                                query_insert = f"""
                                INSERT INTO {table_name} (auction_id, item_id, buyout, bid, quantity, time_left)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """

                                # Execute insertion query with parameters
                                cursor.execute(query_insert, (
                                    auction_id,
                                    item_data.get('id', None),
                                    data.get('buyout', None),
                                    data.get('bid', None),
                                    data.get('quantity', None),
                                    data.get('time_left', None)
                                ))

                                # Confirm changes after processing the file (optional)
                                conn.commit()

                                logging.info(f"Successful insertion for {filename}")
                        else:
                            logging.warning(f"Key 'id' not found or is None in file {filename}")

                    except (json.JSONDecodeError, mysql.connector.Error, KeyError) as e:
                        logging.error(f"Error in file {filename}: {e}")

        # Confirm changes at the end of processing
        conn.commit()

        # Log completion
        logging.info(f"Data migration for table {table_name} completed.")

    except mysql.connector.Error as err:
        # Handle MySQL connection or SQL query execution errors
        logging.error(f"MySQL Error: {err}")

    finally:
        # Close connection and cursor in the 'finally' block to ensure it's closed even if an exception occurs
        cursor.close()
        conn.close()

# Connect to the MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="12792",  # Your password here
    database="AuctionDB"  # Changed to the correct database name
)

# Path to the directory containing JSON files
json_directory = "/home/cmiranda/Documents/ActionsClassicWoW/WOW Sql/05-11-2023"

# Migrate data for the Auctions table
migrate_data(json_directory, "Auctions", conn)

