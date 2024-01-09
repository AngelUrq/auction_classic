import json
import requests
import datetime
import pyodbc
import time
import os
import math


def create_access_token(client_id, client_secret, region='us'):
    data = {'grant_type': 'client_credentials'}
    response = requests.post('https://%s.battle.net/oauth/token' %
                             region, data=data, auth=(client_id, client_secret))
    return response.json()


def retrieve_from_api(config):
    response = create_access_token(config['CLIENT_KEY'], config['SECRET_KEY'])
    token = response['access_token']
    print('Token created')

    response = requests.get('https://us.api.blizzard.com/data/wow/connected-realm/{}/auctions/{}?namespace=dynamic-classic-us&locale=en_US&access_token={}'.format(
        config['connected_realm_id'], config['auction_house_id'], token))

    print('Request done')

    return response.json()


def process_auction(auction):
    rand = None
    if 'rand' in auction['item'].keys():
        rand = str(auction['item']['rand'])

    seed = None
    if 'seed' in auction['item'].keys():
        seed = str(auction['item']['seed'])
    
    bid_gold = auction['bid'] / 10000
    buyout_gold = auction['buyout'] / 10000

    return (auction['id'], auction['item']['id'], bid_gold, buyout_gold, auction['quantity'], auction['time_left'], rand, seed)


def get_auction_data():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    with open(config_path) as json_data:
        config = json.load(json_data)

    data = retrieve_from_api(config)

    auctions = []

    for auction in data['auctions']:
        auctions.append(process_auction(auction))

    print(str(len(auctions)) + ' auctions processed.')
    
    return auctions


def get_item_data():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    with open(config_path) as json_data:
        config = json.load(json_data)

    response = create_access_token(config['CLIENT_KEY'], config['SECRET_KEY'])
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
