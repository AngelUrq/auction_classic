import json
import requests
import datetime
import time
import os
import math


def create_access_token(client_id, client_secret, region='us'):
    data = {'grant_type': 'client_credentials'}
    response = requests.post(
        f'https://{region}.battle.net/oauth/token',
        data=data,
        auth=(client_id, client_secret)
    )
    return response.json()


def process_auction(auction):
    rand = None
    if 'rand' in auction['item'].keys():
        rand = str(auction['item']['rand'])

    seed = None
    if 'seed' in auction['item'].keys():
        seed = str(auction['item']['seed'])
    
    bid_gold = auction['bid'] / 10000.0
    buyout_gold = auction['buyout'] / 10000.0

    return (auction['id'], auction['item']['id'], bid_gold, buyout_gold, auction['quantity'], auction['time_left'], rand, seed)


def get_current_auctions(config):
    response = create_access_token(config['client_key'], config['secret_key'])
    token = response['access_token']
    print('Token created')
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    
    url = (
        'https://us.api.blizzard.com/data/wow/connected-realm/'
        f"{config['realm_id']}/auctions/{config['auction_house_id']}"
        '?namespace=dynamic-classic-us&locale=en_US'
    )
    
    response = requests.get(url, headers=headers)
    print('Request done')
    
    data = response.json()
    
    auctions = []
    for auction in data['auctions']:
        auctions.append(process_auction(auction))
    
    print(f'{len(auctions)} auctions processed.')
    
    return auctions


def get_item_data():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    with open(config_path) as json_data:
        config = json.load(json_data)

    response = create_access_token(config['client_key'], config['secret_key'])
    token = response['access_token']

    result = get_missing_items(config)
    items = []

    for item in result:
        item_id = item[0]
        print('Getting item with id ' + str(item_id))

        try:
            response = requests.get('https://us.api.blizzard.com/data/wow/item/{}?namespace=static-classic-us&locale=en_US&access_token={}'.format(item_id, token))
            
            data = response.json()
            
            item = process_item(data, item_id)
            
            if item is not None:
                items.append(item)
        except Exception as e:
            print(e)


    if len(items) > 0:
        save_items(items, config)
