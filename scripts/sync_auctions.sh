#!/bin/bash

REMOTE_USER="pi"
REMOTE_HOST="10.244.112.103"
REMOTE_DIR="/media/pi/USB-DATA/auctions/"
LOCAL_DIR="/home/angel/source/python/auction_classic/data/auctions_cata/"
SSH_KEY="$HOME/.ssh/id_rsa"

mkdir -p "$LOCAL_DIR"

echo "Hello from the local machine" > "$LOCAL_DIR/hello.txt"

# Sync files using SSH key authentication
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" \
    "$LOCAL_DIR"

if [ $? -eq 0 ]; then
    echo "Sync completed successfully"
else
    echo "Error: Sync failed"
    exit 1
fi
