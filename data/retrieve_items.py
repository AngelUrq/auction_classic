import json
import requests
import datetime
import time
import os
import math

from utils import create_access_token

def process_item(item, item_id):
    name = item['name']
    quality = item['quality']['name']
    level = item['level']
    required_level = item['required_level']
    item_class = item['item_class']['name']
    item_subclass = item['item_subclass']['name']
    purchase_price_gold = math.floor(item['purchase_price'] / 10000)
    purchase_price_silver = math.floor(((item['purchase_price'] / 10000) - purchase_price_gold) * 100)
    sell_price_gold = math.floor(item['sell_price'] / 10000)
    sell_price_silver = math.floor(((item['sell_price'] / 10000) - sell_price_gold) * 100)
    max_count = item['max_count']
    is_equippable = item['is_equippable']
    is_stackable = item['is_stackable']

    if name is not None and quality is not None and level is not None:
        return (item_id, name, quality, level, required_level, item_class, item_subclass, purchase_price_gold, purchase_price_silver, sell_price_gold, sell_price_silver, max_count, is_equippable, is_stackable)


def save_items(items, config):
    print('Loading ' + str(len(items)) + ' items to the database')
    
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+config['db_host'] +';DATABASE='+config['db_name'] +';UID='+config['db_user'] +';PWD='+ config['db_password'] )
    cursor = conn.cursor()

    sql = """INSERT INTO Item (Id, Name, Quality, Level, RequiredLevel, ItemClass, ItemSubClass, PurchasePriceGold, PurchasePriceSilver, SellPriceGold, SellPriceSilver, MaxCount, IsEquippable, IsStackable)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
                
    cursor.executemany(sql, items)

    conn.commit()
    
    cursor.close()
    conn.close()


def get_missing_items(config):
    print('Getting missing items')
    
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+config['db_host'] +';DATABASE='+config['db_name'] +';UID='+config['db_user'] +';PWD='+ config['db_password'] )
    cursor = conn.cursor()

    sql = """SELECT ItemId
            FROM (
                SELECT DISTINCT ItemId FROM Auction
                WHERE ItemId NOT IN (SELECT Id FROM Item)
                ) AS T
            ORDER BY ItemId ASC"""

    cursor.execute(sql)

    result = cursor.fetchall()

    cursor.close()
    conn.close()
    
    return result


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


if __name__ == '__main__':
    get_item_data()
