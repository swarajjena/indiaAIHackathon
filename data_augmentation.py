import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = './data/'

column_to_augment_data = "sub_category"
text_column = "crimeaditionalinfo"

category_to_augment = 'Online Gambling Betting'
number_of_samples_to_be_generated = 1000


def return_list_of_samples(sample_df):
  sample_text = ""
  for (index,sample) in enumerate(sample_df['crimeaditionalinfo'].tolist()):
    sample_text = sample_text + str(index+1)+'. '+sample+'\n---------\n\n'
  return sample_text


OpenAIClient = OpenAI(
  organization='org-mkjQ2vioF8Hd6koCJQavfXmg',
  project='proj_lAbFQNM1q7aFaZPt3cejIxL7',
)

def generate_augmentation(prompt, n=1, model="gpt-4o-mini"):
    """
    Generate augmented text using GPT.
    """
    response = OpenAIClient.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "I am creating a mahcine learning model for predicting the category of reported cyber crime"
                    "You are a helpful assistant for generating synthetic text data. "
                    "Please create variations or new examples of the given text while preserving its meaning."
                    "You can be slightly innovative, while generating"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.7,
        max_tokens=16000,
        n=n,
    )

    return response.choices[0].message.content.split('\n\n')

def generate_prompt(sample_text, augment_samples_per_item):
  return f"""
  Create {augment_samples_per_item} variations of the following text related to '{category_to_augment}':
  Here I have provided couple of samples
  \n\n{sample_text}
  \n\nreturn only list of messages without numbering  divided by newline"
  """

target = 1500

augmentedDF = pd.DataFrame(columns=[text_column, column_to_augment_data])
already_processed = 0


print("Augmenting data for: ",category_to_augment)

train_df = pd.read_csv(DATA_DIR + 'train.csv')
train_df['sub_category'] = train_df['sub_category'].replace("", None)  # Replace blanks with None for consistency
train_df['sub_category'] = train_df['sub_category'].fillna(train_df['category'])



samples_to_send_in_each_batch = 10

for i in range(0,number_of_samples_to_be_generated ,samples_to_send_in_each_batch):
  augmented_rows = []
  print(i,'/',number_of_samples_to_be_generated)

  category_df = train_df[train_df[column_to_augment_data]==category_to_augment]

  sample_df = category_df.iloc[i:i+samples_to_send_in_each_batch]

  prompt = generate_prompt(return_list_of_samples(sample_df), number_of_samples_to_be_generated*samples_to_send_in_each_batch)

  attempts = 0
  while(attempts <3):
    augmented_texts = generate_augmentation(prompt)

    if(len(augmented_texts)>1):
      break
    attempts = attempts + 1

  print(len(augmented_texts))

  for augmented_text in augmented_texts[0:]:
    if(len(augmented_text)>10):
      augmented_rows.append({text_column: augmented_text, column_to_augment_data: category_to_augment})

  augmentedDF = pd.concat([augmentedDF, pd.DataFrame(augmented_rows)]).reindex()
  already_processed = i



augmentedDF = augmentedDF[~augmentedDF['crimeaditionalinfo'].str.contains("I can't assist")]

augment_data_file = DATA_DIR + 'augmented_'+column_to_augment_data+'.csv'
existingAugmentedDF = pd.read_csv(augment_data_file)

pd.concat([existingAugmentedDF, augmentedDF])[['crimeaditionalinfo','sub_category']].reindex().to_csv(augment_data_file,index=False)

existingAugmentedDF = pd.read_csv(augment_data_file)

print(existingAugmentedDF.sub_category.value_counts())

