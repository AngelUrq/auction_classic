#!/bin/bash

REMOTE_USER="pi"
REMOTE_HOST="192.168.100.194"
REMOTE_AUCTIONS_DIR="/media/pi/USB-DATA/auctions/"
REMOTE_COMMODITIES_DIR="/media/pi/USB-DATA/commodities/"
LOCAL_AUCTIONS_DIR="/home/angel/source/auction_classic/data/auctions/"
LOCAL_COMMODITIES_DIR="/home/angel/source/auction_classic/data/commodities/"
SSH_KEY="$HOME/.ssh/id_rsa"

mkdir -p "$LOCAL_AUCTIONS_DIR"
mkdir -p "$LOCAL_COMMODITIES_DIR"

rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_AUCTIONS_DIR" \
    "$LOCAL_AUCTIONS_DIR"

echo "Skipping commodities sync"

if [ $? -eq 0 ]; then
    echo "Sync completed successfully"
else
    echo "Error: Sync failed"
    exit 1
fi
