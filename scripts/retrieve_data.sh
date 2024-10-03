#!/bin/bash          

echo "Retrieving token"

client_key=""
secret_key=""
realm_id="4408"
auction_house_id="2"

token=`curl -u $client_key:$secret_key -d grant_type=client_credentials https://us.battle.net/oauth/token | jq -r '.access_token'`

data_dir="data"

mkdir -p $data_dir/"$(date +"%d-%m-%Y")"

echo "Downloading auction data..."

curl "https://us.api.blizzard.com/data/wow/connected-realm/$realm_id/auctions/$auction_house_id?namespace=dynamic-classic-us&locale=en_US&access_token=$token" -o $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).json

echo "Data downloaded"

for i in {1..2}
do
    hash=`md5sum $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).json | awk '{print $1}'`

    echo "Sleeping for 10 minutes..."
    sleep 600

    echo "Download again..."
    curl "https://us.api.blizzard.com/data/wow/connected-realm/$realm_id/auctions/$auction_house_id?namespace=dynamic-classic-us&locale=en_US&access_token=$token" -o $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).copy.json

    hash_copy=`md5sum $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).copy.json | awk '{print $1}'`

    if [ "$hash" == "$hash_copy" ]; then
        echo "Files are the same"
        rm $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).copy.json
    else
        echo "Files are different"
        rm $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).json
        mv $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).copy.json $data_dir/"$(date +"%d-%m-%Y")"/$(date -d "today" +%Y%m%dT%H).json
    fi
done

echo "Done!"
