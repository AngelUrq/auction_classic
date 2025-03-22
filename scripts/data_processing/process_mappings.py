import os, json, argparse
from datetime import datetime
from tqdm import tqdm

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

    # Padding: 0, Unknown: 1
    item_to_index = {0: 0, 1: 1}
    context_to_index = {0: 0, 1: 1}
    bonus_id_to_index = {0: 0, 1: 1} 
    modifier_type_to_index = {0: 0, 1: 1}

    for idx, item_id in enumerate(sorted(item_ids), start=2):
        item_to_index[item_id] = idx
        
    for idx, context in enumerate(sorted(contexts), start=2):
        context_to_index[context] = idx
        
    for idx, bonus_id in enumerate(sorted(bonus_ids), start=2):
        bonus_id_to_index[bonus_id] = idx
        
    for idx, mod_type in enumerate(sorted(modifier_types), start=2):
        modifier_type_to_index[mod_type] = idx

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
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the auctions folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output the mapping files')
    args = parser.parse_args()
    
    process_mappings(args)
    