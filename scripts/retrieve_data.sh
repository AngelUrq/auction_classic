#!/bin/bash

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Configuration
client_key="c39078bd5f0f4e798a3a1b734dd9d280"
secret_key="4UEplhA8jYa8wvX58C5QdV7JDTDY9rNX"
realm_id="4408"
auction_house_id="6"
data_dir="data"

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check HTTP response
check_response() {
    local response_code=$1
    local url=$2
    if [ $response_code -ne 200 ]; then
        log_message "ERROR: Failed to fetch from $url (HTTP $response_code)"
        return 1
    fi
    return 0
}

# Function to get OAuth token
get_token() {
    if [[ -z "$client_key" || -z "$secret_key" ]]; then
        log_message "ERROR: Client key or secret key not set"
        exit 1
    }
    
    local response=$(curl -s -w "%{http_code}" -u "$client_key:$secret_key" \
        -d grant_type=client_credentials \
        https://us.battle.net/oauth/token)
    
    local http_code=${response: -3}
    local body=${response:0:${#response}-3}
    
    if ! check_response "$http_code" "oauth/token"; then
        log_message "ERROR: Failed to obtain OAuth token"
        exit 1
    }
    
    echo "$body" | jq -r '.access_token'
}

# Function to download auction data
download_auction_data() {
    local token=$1
    local output_file=$2
    
    local response=$(curl -s -w "%{http_code}" \
        "https://us.api.blizzard.com/data/wow/connected-realm/$realm_id/auctions/$auction_house_id?namespace=dynamic-classic-us&locale=en_US&access_token=$token" \
        -o "$output_file")
    
    if ! check_response "$response" "auctions API"; then
        return 1
    fi
    
    # Validate JSON format
    if ! jq empty "$output_file" 2>/dev/null; then
        log_message "ERROR: Invalid JSON received"
        return 1
    }
    
    return 0
}

# Create directory structure
year=$(date +"%Y")
month=$(date +"%m")
day=$(date +"%d")
daily_dir="$data_dir/$year/$month/$day"
mkdir -p "$daily_dir"

# Main execution
log_message "Starting auction data collection"

# Get OAuth token
log_message "Retrieving OAuth token"
token=$(get_token)

# Initial download
current_file="$daily_dir/$(date -d "today" +%Y%m%dT%H).json"
log_message "Downloading initial auction data..."
if ! download_auction_data "$token" "$current_file"; then
    log_message "ERROR: Failed to download initial auction data"
    exit 1
fi
log_message "Initial data downloaded successfully"

# Check for updates
for i in {1..2}; do
    log_message "Starting update check $i of 2"
    
    # Calculate hash of original file
    hash=$(md5sum "$current_file" | awk '{print $1}')
    
    log_message "Waiting 10 minutes before next check..."
    sleep 600
    
    # Download new copy
    temp_file="$daily_dir/$(date -d "today" +%Y%m%dT%H).copy.json"
    log_message "Downloading new data..."
    
    if ! download_auction_data "$token" "$temp_file"; then
        log_message "ERROR: Failed to download update $i"
        rm -f "$temp_file"
        continue
    fi
    
    # Calculate hash of new copy
    hash_copy=$(md5sum "$temp_file" | awk '{print $1}')
    
    # Compare files
    if [ "$hash" == "$hash_copy" ]; then
        log_message "No changes detected in auction data"
        rm -f "$temp_file"
    else
        log_message "Changes detected - updating data file"
        rm -f "$current_file"
        mv "$temp_file" "$current_file"
    fi
done

log_message "Auction data collection completed"