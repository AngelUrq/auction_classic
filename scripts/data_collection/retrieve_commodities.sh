#!/bin/bash

echo "Retrieving token"

# Configuration
client_key=""
secret_key=""

# Get OAuth token
token=`curl -u $client_key:$secret_key -d grant_type=client_credentials https://us.battle.net/oauth/token | jq -r '.access_token'`

# Create directory structure using yyyy/mm/dd format
data_dir="/media/pi/USB-DATA/commodities"
year=$(date +"%Y")
month=$(date +"%m")
day=$(date +"%d")
daily_dir="$data_dir/$year/$month/$day"
mkdir -p "$daily_dir"

echo "Downloading commodity data..."
# Download initial data
curl "https://us.api.blizzard.com/data/wow/auctions/commodities?namespace=dynamic-us&locale=en_US" \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/json" \
    -o "${daily_dir}/$(date -d "today" +%Y%m%dT%H).json"

echo "Data downloaded"

# Check for updates
for i in {1..2}
do
    # Calculate hash of original file
    hash=`md5sum "$daily_dir/$(date -d "today" +%Y%m%dT%H).json" | awk '{print $1}'`
    
    echo "Sleeping for 10 minutes..."
    sleep 600
    
    echo "Download again..."
    # Download new copy
    curl "https://us.api.blizzard.com/data/wow/auctions/commodities?namespace=dynamic-us&locale=en_US" \
        -H "Authorization: Bearer ${token}" \
        -H "Accept: application/json" \
        -o "${daily_dir}/$(date -d "today" +%Y%m%dT%H).copy.json"
    
    # Calculate hash of new copy
    hash_copy=`md5sum "$daily_dir/$(date -d "today" +%Y%m%dT%H).copy.json" | awk '{print $1}'`
    
    # Compare files
    if [ "$hash" == "$hash_copy" ]; then
        echo "Files are the same"
        rm "$daily_dir/$(date -d "today" +%Y%m%dT%H).copy.json"
    else
        echo "Files are different"
        rm "$daily_dir/$(date -d "today" +%Y%m%dT%H).json"
        mv "$daily_dir/$(date -d "today" +%Y%m%dT%H).copy.json" "$daily_dir/$(date -d "today" +%Y%m%dT%H).json"
    fi
done

echo "Done!" 