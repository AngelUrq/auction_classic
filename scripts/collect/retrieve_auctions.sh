#!/bin/bash

echo "Retrieving token..."

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/../../.env"

if [ -f "$env_file" ]; then
    set -a
    source "$env_file"
    set +a
fi

client_key="${BLIZZARD_CLIENT_KEY}"
secret_key="${BLIZZARD_SECRET_KEY}"
realm_id="${BLIZZARD_REALM_ID}"

if [ -z "$client_key" ] || [ -z "$secret_key" ]; then
    echo "Error: BLIZZARD_CLIENT_KEY and BLIZZARD_SECRET_KEY must be set" >&2
    exit 1
fi

# Get OAuth token
token=`curl -u $client_key:$secret_key -d grant_type=client_credentials https://${BLIZZARD_REGION}.battle.net/oauth/token | jq -r '.access_token'`
echo "Token retrieved"

if [ "$token" = "null" ]; then
    echo "Error: Token is null" >&2
    exit 1
fi

# Create directory structure using yyyy/mm/dd format
data_dir="${COLLECTOR_AUCTIONS_DIR}"
year=$(date +"%Y")
month=$(date +"%m")
day=$(date +"%d")
daily_dir="$data_dir/$year/$month/$day"
mkdir -p "$daily_dir"

output_file="${daily_dir}/$(date +%Y%m%dT%H).json"
min_size_bytes=1048576  # 1 MB
max_retries=3

for attempt in $(seq 1 $max_retries); do
    echo "Downloading auction data (attempt $attempt)..."
    curl "https://${BLIZZARD_REGION}.api.blizzard.com/data/wow/connected-realm/${realm_id}/auctions?namespace=${BLIZZARD_NAMESPACE}&locale=${BLIZZARD_LOCALE}" \
        -H "Authorization: Bearer ${token}" \
        -H "Accept: application/json" \
        -o "$output_file"

    curl_exit=$?
    file_size=$(wc -c < "$output_file" 2>/dev/null || echo 0)

    if [ "$curl_exit" -eq 0 ] && [ "$file_size" -ge "$min_size_bytes" ] && jq empty "$output_file" 2>/dev/null; then
        echo "Download successful ($file_size bytes)"
        break
    fi

    echo "Download failed (curl: $curl_exit, size: $file_size bytes), retrying..."
    rm -f "$output_file"

    if [ "$attempt" -eq "$max_retries" ]; then
        echo "Error: All $max_retries attempts failed" >&2
        exit 1
    fi
done

echo "Syncing to server..."
rsync -az --remove-source-files -e "ssh -i $HOME/.ssh/id_rsa" "$data_dir/" "$SERVER_USER@$SERVER_HOST:$SERVER_AUCTIONS_DIR/"
find "$data_dir" -type d -empty -delete
echo "Done!"
