#!/bin/bash

# Configuration
EXPECTED_FILES=24
COMMODITIES_BASE="/media/pi/USB-DATA/commodities"
AUCTIONS_BASE="/media/pi/USB-DATA/auctions"
NOTIFICATION_CHANNEL="goblin-alerts"

# Get yesterday's date components
YEAR=$(date -d "yesterday" +"%Y")
MONTH=$(date -d "yesterday" +"%m")
DAY=$(date -d "yesterday" +"%d")

# Function to check if JSON is valid
validate_json() {
    if jq empty "$1" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to verify directory data
verify_directory() {
    local dir_path="$1"
    local dir_type="$2"
    local report=""
    local error_count=0
    local valid_count=0
    local missing_hours=""

    # Check if directory exists
    if [ ! -d "$dir_path" ]; then
        echo "🚫 $dir_type directory for $YEAR/$MONTH/$DAY not found!"
        curl -d "$dir_type directory for $YEAR/$MONTH/$DAY not found! 🚫" ntfy.sh/$NOTIFICATION_CHANNEL
        return 1
    fi

    # Check each hour's file
    for hour in {00..23}; do
        file_path="$dir_path/${YEAR}${MONTH}${DAY}T${hour}.json"
        
        if [ ! -f "$file_path" ]; then
            missing_hours="$missing_hours $hour:00"
            ((error_count++))
            continue
        fi

        # Check file size (minimum 100 bytes)
        if [ $(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path") -lt 100 ]; then
            report="$report\n⚠️ Hour $hour:00 file is too small"
            ((error_count++))
            continue
        fi

        # Validate JSON
        if ! validate_json "$file_path"; then
            report="$report\n⚠️ Hour $hour:00 has invalid JSON"
            ((error_count++))
            continue
        fi

        ((valid_count++))
    done

    # Prepare and send report
    local status_icon="✅"
    local status_msg="OK"
    
    if [ $error_count -gt 0 ]; then
        status_icon="⚠️"
        status_msg="WARNING"
    fi
    if [ $valid_count -eq 0 ]; then
        status_icon="🚫"
        status_msg="CRITICAL"
    fi

    local summary="$status_icon $dir_type Data Report ($YEAR/$MONTH/$DAY):\n"
    summary+="Status: $status_msg\n"
    summary+="Valid files: $valid_count/$EXPECTED_FILES\n"
    
    if [ ! -z "$missing_hours" ]; then
        summary+="Missing hours:$missing_hours\n"
    fi
    if [ ! -z "$report" ]; then
        summary+="Issues:$report"
    fi

    echo -e "$summary"
    curl -d "$summary" ntfy.sh/$NOTIFICATION_CHANNEL
}

echo "Starting daily data verification for $YEAR/$MONTH/$DAY"

commodities_path="$COMMODITIES_BASE/$YEAR/$MONTH/$DAY"
verify_directory "$commodities_path" "Commodities"

auctions_path="$AUCTIONS_BASE/$YEAR/$MONTH/$DAY"
verify_directory "$auctions_path" "Auctions"

echo "Verification complete!" 