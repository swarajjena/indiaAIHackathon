import json
DATA_DIR = './data/'

def generate_mapping(df):
    category_mapping = {category: idx for idx, category in enumerate(df['category'].unique())}
    subcategory_mapping = {subcategory: idx for idx, subcategory in enumerate(df['sub_category'].unique())}
    
    return {
        "category_mapping": category_mapping, 
        "subcategory_mapping": subcategory_mapping,
        "no_of_categories": len(category_mapping),
        "no_of_subcategories": len(subcategory_mapping)        
    }

def save_mappings(mappings):
    with open(DATA_DIR+'mappings.json', 'w') as f:
        json.dump(mappings, f)


def load_mappings():
    with open(DATA_DIR+'mappings.json', 'r') as f:
        mappings = json.load(f)
        
    return mappings