import os
import json
import csv

# Set the path to the auctions folder
auctions_folder = 'auctions'

# Create a list to store the results
results = []

# Iterate over each day folder
for day_folder in os.listdir(auctions_folder):
    day_folder_path = os.path.join(auctions_folder, day_folder)
    if os.path.isdir(day_folder_path):
        # Extract the date from the folder name
        date = day_folder

        # Convert the date to YYYYMMDD format
        day, month, year = date.split('-')
        date_yyyymmdd = f'{year}{month}{day}'

        # Iterate over each hour of the day
        for hour in range(24):
            hour_str = f'{hour:02d}'
            json_file_name = f'{date_yyyymmdd}T{hour_str}.json'
            json_file_path = os.path.join(day_folder_path, json_file_name)

            # Check if the JSON file exists
            if os.path.exists(json_file_path):
                try:
                    # Try to load the JSON file
                    with open(json_file_path, 'r') as json_file:
                        json.load(json_file)
                    json_valid = 'Valid'
                except json.JSONDecodeError:
                    json_valid = 'Invalid'
            else:
                json_valid = 'Missing'

            # Append the result to the list
            results.append([date, hour_str, json_valid])

# Sort the results by date
results.sort(key=lambda x: (x[0].split('-')[2], x[0].split('-')[1], x[0].split('-')[0]))

# Create a CSV writer to report the results
with open('data_report.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Hour', 'Status'])
    writer.writerows(results)
