import torch 

# Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, category_labels, subcategory_labels):
        self.encodings = encodings
        self.category_labels = category_labels
        self.subcategory_labels = subcategory_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.category_labels[idx]).long()
        return item

    def __len__(self):
        return len(self.category_labels)
