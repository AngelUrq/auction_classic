import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

def find_max_bonuses_modifiers(args):
    print('Finding maximum number of bonuses and modifiers...')
    
    data_dir = args.data_dir
    file_info = {}
    
    # Collect all JSON files and sort by date
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                try:
                    date = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')
                    file_info[filepath] = date
                except ValueError:
                    print(f"Skipping file with invalid date format: {filename}")
                    continue

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}
    
    total_files = len(file_info)
    print(f'Processing {total_files} JSON files')

    max_bonuses = 0
    max_modifiers = 0
    max_bonuses_file = ""
    max_modifiers_file = ""
    max_bonuses_item = ""
    max_modifiers_item = ""

    for filepath in tqdm(list(file_info.keys())):
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading file {filepath}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error loading file {filepath}: {e}")
            continue
            
        auctions = json_data.get('auctions', [])
        
        for auction in auctions:
            item = auction.get('item', {})
            
            # Count bonuses
            bonus_lists = item.get('bonus_lists', [])
            num_bonuses = len(bonus_lists)
            
            if num_bonuses > max_bonuses:
                max_bonuses = num_bonuses
                max_bonuses_file = filepath
                max_bonuses_item = item.get('id', 'unknown')
                print(f"New max bonuses: {max_bonuses} in file {os.path.basename(filepath)} for item {max_bonuses_item}")
            
            # Count modifiers
            modifiers = item.get('modifiers', [])
            num_modifiers = len(modifiers)
            
            if num_modifiers > max_modifiers:
                max_modifiers = num_modifiers
                max_modifiers_file = filepath
                max_modifiers_item = item.get('id', 'unknown')
                print(f"New max modifiers: {max_modifiers} in file {os.path.basename(filepath)} for item {max_modifiers_item}")

    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f"Maximum number of bonuses: {max_bonuses}")
    print(f"Found in file: {os.path.basename(max_bonuses_file)}")
    print(f"Item ID: {max_bonuses_item}")
    print()
    print(f"Maximum number of modifiers: {max_modifiers}")
    print(f"Found in file: {os.path.basename(max_modifiers_file)}")
    print(f"Item ID: {max_modifiers_item}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find maximum number of bonuses and modifiers in auction JSON files')
    parser.add_argument('--data_dir', type=str, help='Path to the auctions folder', required=True)
    
    args = parser.parse_args()
    find_max_bonuses_modifiers(args)
