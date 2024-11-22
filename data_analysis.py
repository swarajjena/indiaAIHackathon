import pandas as pd

DATA_DIR = './data/'


train_df = pd.read_csv(DATA_DIR + 'train.csv')
test_df = pd.read_csv(DATA_DIR + 'test.csv')

print('Total Data in Training Dataset:',train_df.shape[0])
print('Total Data in Testing Dataset:',test_df.shape[0])

print("Total Categories in Training Datasset:",train_df.category.nunique())
print("Total Sub-Categories in Training Datasset:",train_df.sub_category.nunique())

