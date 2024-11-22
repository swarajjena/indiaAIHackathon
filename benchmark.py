import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score
from utils.clean_dataset import clean_dataset
from utils.category_mapping import generate_mapping,save_mappings
from utils.prepare_dataset import prepare_dataset

login()


DATA_DIR = './data/'

# Read Data
train_df = pd.read_csv(DATA_DIR + 'train.csv')
test_df = pd.read_csv(DATA_DIR + 'test.csv')

train_df = clean_dataset(train_df)
test_df = clean_dataset(test_df)

mappings = generate_mapping(train_df)
save_mappings(mappings)

category_mapping = mappings["category_mapping"]
subcategory_mapping = mappings["subcategory_mapping"]
num_category_labels = mappings["no_of_categories"]
num_subcategory_labels = mappings["no_of_subcategories"]


# Map categories and subcategories to numerical labels
train_df['category_label'] = train_df['category'].map(category_mapping)
train_df['subcategory_label'] = train_df['sub_category'].map(subcategory_mapping)

test_df['category_label'] = test_df['category'].map(category_mapping)
test_df['subcategory_label'] = test_df['sub_category'].map(subcategory_mapping)


# If some category or subcategory not present in training dataset
test_df = test_df.dropna(subset=['category_label', 'subcategory_label'])

# Model Selection
model_checkpoint = "bert-base-uncased"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Prepare the dataset for training
train_set = prepare_dataset(train_df, tokenizer)
test_set = prepare_dataset(test_df, tokenizer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_subcategory_labels,  # Number of labels for category classification
)

model.resize_token_embeddings(len(tokenizer))


# Set Trainable Parameters
for param in model.base_model.parameters():  # Freeze all layers of the base model
    param.requires_grad = False

for i in range(1):
  for param in model.base_model.encoder.layer[-1*(i+1)].parameters():
      param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

model.base_model.config.pad_token_id = model.base_model.config.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# Define a compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "f1_score":f1_score(labels, predictions, average='weighted')}



# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=64,  # Adjust batch size for Mistral's memory usage
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,  # Enable mixed precision for A100
    # save_strategy="no",
    max_grad_norm=1.0,  # Clip gradients
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics,  # Custom metrics function
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(DATA_DIR + 'benchmark-'+model_checkpoint)