from utils.custom_dataset import CustomDataset 

def tokenize_function(df, tokenizer):
    return tokenizer(df['crimeaditionalinfo'].tolist(), padding="max_length", truncation=True, max_length=128,return_tensors="pt")

def prepare_dataset(df, tokenizer):
    encodings = tokenize_function(df, tokenizer)

    encodings['category_labels'] = list(df['category_label'])
    encodings['subcategory_label'] = list(df['subcategory_label'])

    return CustomDataset(encodings, encodings['category_labels'], encodings['subcategory_label'])