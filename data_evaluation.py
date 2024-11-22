from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import login
from utils.custom_dataset import CustomDataset 
from utils.category_mapping import load_mappings
import torch
from sklearn.metrics import accuracy_score, f1_score
from utils.clean_dataset import clean_dataset
import pandas as pd
from utils.prepare_dataset import prepare_dataset
import numpy as np
from dotenv import load_dotenv
load_dotenv()

login()

DATA_DIR = './data/'
model_checkpoint = DATA_DIR + "benchmark-bert-base-uncased"

# Load Mappings
mappings = load_mappings()
category_mapping = mappings["category_mapping"]
subcategory_mapping = mappings["subcategory_mapping"]
num_category_labels = mappings["no_of_categories"]
num_subcategory_labels = mappings["no_of_subcategories"]


test_df = pd.read_csv(DATA_DIR + 'test.csv')

test_df = clean_dataset(test_df)
test_df['category_label'] = test_df['category'].map(category_mapping)
test_df['subcategory_label'] = test_df['sub_category'].map(subcategory_mapping)
test_df = test_df.dropna(subset=['category_label', 'subcategory_label'])

tokeniser_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_checkpoint)

test_set = prepare_dataset(test_df, tokenizer)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_subcategory_labels,  # Number of labels for category classification
)

# Define a compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "f1_score":f1_score(labels, predictions, average='weighted')}


training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=128,
)

trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)


logits = trainer.predict(test_set)
test_df["predicted_label_index"] = np.argmax(logits.predictions, 1)

index_category_mapping= {}
for index,category in enumerate(subcategory_mapping):
  index_category_mapping[index] = category

test_df["predicted_label"] = test_df["predicted_label_index"].map(lambda x: index_category_mapping[x])

test_df['correct_prediction'] = test_df['subcategory_label'] == test_df['predicted_label_index']

# Calculate the percentage of correct predictions for each category
accuracy_by_category = test_df.groupby('sub_category')['correct_prediction'].mean() * 100

# Convert the result into a DataFrame for better visualization
accuracy_df = accuracy_by_category.reset_index()
accuracy_df.rename(columns={'correct_prediction': 'accuracy_percentage'}, inplace=True)

category_df = test_df.sub_category.value_counts()
category_df = pd.DataFrame({'sub_category': category_df.index, 'count': category_df.values})

merged_df = pd.merge(category_df, accuracy_df, on='sub_category', how='left')
merged_df["correctly_predicted"] = merged_df["count"] * merged_df["accuracy_percentage"] /100
merged_df["incorrectly_predicted"] = merged_df["count"] - merged_df["correctly_predicted"]

