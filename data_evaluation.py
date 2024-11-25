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
import os
import argparse

load_dotenv()

hf_token = os.getenv("HUGGING_FACE_TOKEN")
login(token=hf_token)

DATA_DIR = './data/'


model_checkpoint = "swarajjena/benchmark-xlm-roberta-base"

# Load Mappings
mappings = load_mappings()
category_mapping = mappings["category_mapping"]
subcategory_mapping = mappings["subcategory_mapping"]
num_category_labels = mappings["no_of_categories"]
num_subcategory_labels = mappings["no_of_subcategories"]

tokeniser_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_checkpoint)


# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_category_labels,  # Number of labels for category classification
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=128,
)

# Define a compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "f1_score":f1_score(labels, predictions, average='weighted')}


trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

index_category_mapping= {}
for index,category in enumerate(category_mapping):
    index_category_mapping[index] = category





def evaluate_accuracy(filename = 'test.csv'):
    test_df = pd.read_csv(DATA_DIR + filename)

    test_df = clean_dataset(test_df)
    test_df['category_label'] = test_df['category'].map(category_mapping)
    test_df['subcategory_label'] = test_df['sub_category'].map(subcategory_mapping)
    test_df = test_df.dropna(subset=['category_label', 'subcategory_label'])


    test_set = prepare_dataset(test_df, tokenizer)

    logits = trainer.predict(test_set)
    test_df["predicted_label_index"] = np.argmax(logits.predictions, 1)

    test_df["predicted_label"] = test_df["predicted_label_index"].map(lambda x: index_category_mapping[x])
    test_df['correct_prediction'] = test_df['category_label'] == test_df['predicted_label_index']


    accuracy = 100*accuracy_score(test_df['category_label'], test_df['predicted_label_index'])
    print(f"\n\nAccuracy: {accuracy:.2f}%")

    f1_score_value = 100*f1_score(test_df['category_label'], test_df['predicted_label_index'], average='weighted')
    print(f"F1 Score: {f1_score_value:.2f}%\n\n")

    # Calculate the percentage of correct predictions for each category
    accuracy_by_category = test_df.groupby('category')['correct_prediction'].mean() * 100

    # Convert the result into a DataFrame for better visualization
    accuracy_df = accuracy_by_category.reset_index()
    accuracy_df.rename(columns={'correct_prediction': 'accuracy_percentage'}, inplace=True)

    category_df = test_df.category.value_counts()
    category_df = pd.DataFrame({'category': category_df.index, 'count': category_df.values})

    merged_df = pd.merge(category_df, accuracy_df, on='category', how='left')
    merged_df["correctly_predicted"] = merged_df["count"] * merged_df["accuracy_percentage"] /100
    merged_df["incorrectly_predicted"] = merged_df["count"] - merged_df["correctly_predicted"]

    print(merged_df)

def main():
    default_path = "test.csv"  # Set your default file path
    file_path = input(f"Please enter the path to the csv file wrt ./data folder you want to evaluate: (default: {default_path}): ") or default_path

    try:
        with open(file_path, 'r') as file:
            print("Evaluating File", file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Call the evaluate_accuracy function
    evaluate_accuracy(file_path)

if __name__ == "__main__":
    main()