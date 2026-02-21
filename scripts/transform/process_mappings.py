import os, json, argparse
from datetime import datetime
from tqdm import tqdm


def update_mapping(output_dir, filename, new_items, default_map=None):
    if default_map is None:
        default_map = {"0": 0, "1": 1}
        
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            mapping = json.load(f)
    else:
        mapping = default_map.copy()
        
    current_idx = max(mapping.values()) + 1
    existing_keys = set(mapping.keys())
    
    for item in sorted(new_items):
        item_str = str(item)
        if item_str not in existing_keys:
            mapping[item_str] = current_idx
            current_idx += 1
            
    return mapping

def process_mappings(args):
    print('Processing auctions...')
    file_info = {}
    data_dir = args.data_dir

    item_ids = set()
    contexts = set() 
    bonus_ids = set()
    modifier_types = set()

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            date = datetime.strptime(filename.split('.')[0], '%Y%m%dT%H')
            file_info[filepath] = date

    file_info = {k: v for k, v in sorted(file_info.items(), key=lambda item: item[1])}

    for filepath in tqdm(list(file_info.keys())):
        with open(filepath, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading file {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error loading file {filepath}: {e}")
                continue
                
        auctions = json_data['auctions']
        
        for auction in auctions:
            item = auction['item']
            item_ids.add(item['id'])
            
            if 'context' in item:
                contexts.add(item['context'])
                
            if 'bonus_lists' in item:
                for bonus_id in item['bonus_lists']:
                    bonus_ids.add(bonus_id)
                
            if 'modifiers' in item:
                modifier_types.update(mod['type'] for mod in item['modifiers'])

    item_to_index = update_mapping(args.output_dir, 'item_to_idx.json', item_ids)
    context_to_index = update_mapping(args.output_dir, 'context_to_idx.json', contexts)
    bonus_id_to_index = update_mapping(args.output_dir, 'bonus_to_idx.json', bonus_ids)
    modifier_type_to_index = update_mapping(args.output_dir, 'modtype_to_idx.json', modifier_types)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'item_to_idx.json'), 'w') as f:
        json.dump(item_to_index, f, indent=4)
        
    with open(os.path.join(args.output_dir, 'context_to_idx.json'), 'w') as f:
        json.dump(context_to_index, f, indent=4)
        
    with open(os.path.join(args.output_dir, 'bonus_to_idx.json'), 'w') as f:
        json.dump(bonus_id_to_index, f, indent=4)
        
    with open(os.path.join(args.output_dir, 'modtype_to_idx.json'), 'w') as f:
        json.dump(modifier_type_to_index, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mapping files from auction data')
    parser.add_argument('--data_dir', type=str, default='data/tww/auctions/', help='Path to the auctions folder')
    parser.add_argument('--output_dir', type=str, default='generated/mappings/', help='Path to output the mapping files')
    args = parser.parse_args()

    mapping_files = ['item_to_idx.json', 'context_to_idx.json', 'bonus_to_idx.json', 'modtype_to_idx.json']
    if all(os.path.exists(os.path.join(args.output_dir, f)) for f in mapping_files):
        print(f"Skipping: mapping files already exist in {args.output_dir}")
        exit(0)

    process_mappings(args)
    