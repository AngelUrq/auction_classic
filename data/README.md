# Auction House Data Retrieval Script

This Bash script is designed to retrieve auction house data from the World of Warcraft API every hour and store the data in a JSON file. The retrieved data includes information about items currently being sold on the auction house, such as `auctionId`, `itemId`, `buyout`, `bid`, and `quantity`. To automate the data retrieval, you can use `crontab` to run the script at regular intervals.

## Prerequisites

Before using this script, make sure you have the following prerequisites in place:

- **World of Warcraft API Access**: You need to have access to the World of Warcraft API. You can obtain an API key from the [Blizzard Developer Portal](https://develop.battle.net/).

- **jq**: Ensure that the `jq` command-line tool is installed on your system. You can install it using your package manager (e.g., `apt-get`, `brew`, `yum`).

- **Cron Job**: You should be familiar with setting up cron jobs on your system to schedule the script's execution.

## Usage

1. Clone or download this repository to your local machine.

2. Open the `retrieve_data.sh` script in a text editor of your choice.

3. Update the following variables in the script with your own information:

   - `client_key`: Replace with your World of Warcraft API key.
   - `secret_key`: Replace with your World of Warcraft API key.
   - `realm_id`: Set the realm ID you want to retrieve data for.
   - `auction_house_id`: Set the auction house ID.
   - `data_dir`: Set the directory where you want to store the JSON file.

4. Save the script with your changes.

5. Set up a `cron` job to run the script every hour. Open your terminal and run the following command to edit your crontab:

   ```bash
   crontab -e
   ```

Now, the script will automatically run every hour, retrieve auction house data, and save it in the specified JSON file.

Sure, here's an additional section to explain how to run the script manually:

## Manual Execution

If you wish to run the script manually to retrieve auction house data before the scheduled hourly execution, follow these steps:

1. Open a terminal window.

2. Navigate to the directory where you've saved the `retrieve_data.sh` script.

3. Make sure the script has the necessary permissions to execute. If it doesn't, you can grant execute permission by running the following command:

   ```bash
   chmod +x retrieve_data.sh
   ```

4. Run the script by executing the following command:

   ```bash
   ./retrieve_data.sh
   ```

5. The script will initiate a request to the World of Warcraft API, retrieve auction house data, and save it to the specified JSON file.

6. Once the script has finished running, you can check the JSON file to view the retrieved auction data.

   ```bash
   cat /path/to/your/output_file.json
   ```

Make sure to replace `/path/to/your/output_file.json` with the actual path to your JSON file.

Running the script manually can be useful for testing, troubleshooting, or retrieving auction house data outside the scheduled cron job.

## Script Details

- The script makes a request to the World of Warcraft API using your API key to retrieve auction house data for the specified realm and faction.

- It stores the retrieved data in a JSON file in the specified location. The data includes the fields `auctionId`, `itemId`, `buyout`, `bid`, and `quantity` for each item being sold. The data retrieved from the World of Warcraft API contains the following fields, each providing information about an item currently listed on the auction house:

1. `id`: A unique identifier for each auction listing. This ID is used to differentiate between different auctions and track specific listings.
2. `item_id`: The identifier of the item being sold. This corresponds to a specific in-game item, and you can use this ID to look up item details like name, quality, and item level.
3. `buyout`: The "buyout" price for the item, which represents the fixed price at which a buyer can immediately purchase the item and end the auction. This is often expressed in in-game currency (e.g., gold, silver, copper).
4. `bid`: The current bid amount for the item, which represents the minimum amount a bidder needs to offer to participate in the auction. Bidders can place higher bids to compete with other potential buyers.
5. `quantity`: The number of items available in the auction listing. It indicates how many instances of the item are being sold within the same auction.
6. `time_left`: Time left in auction. An expired item is send back to the seller.

These fields are crucial for understanding and evaluating the current status of items on the auction house, and they provide valuable information for players and developers looking to interact with the World of Warcraft economy.

- The script uses the `jq` tool to process and format the JSON data.
- The JSON data is appended to the output file with a timestamp, creating a historical record of auction house data.

**Note**: Please be aware of any rate limits or usage restrictions imposed by the World of Warcraft API. Make sure to use the API key responsibly and in accordance with Blizzard's terms of service.
