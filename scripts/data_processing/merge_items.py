import pandas as pd

items_wotlk = pd.read_csv('../data/items_wotlk.csv')
items_cata = pd.read_csv('../data/items_cata.csv')

print(f'Length of items_wotlk: {len(items_wotlk)}')
print(f'Length of items_cata: {len(items_cata)}')

items = pd.concat([items_wotlk, items_cata])
items = items.drop_duplicates(subset=['item_id'])
items.set_index('item_id')

print(f'Length of items: {len(items)}')

items.to_csv('../data/items.csv', index=False)
