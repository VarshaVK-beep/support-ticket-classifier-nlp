from datasets import load_dataset
import pandas as pd

dataset = load_dataset('Tobi-Bueck/customer-support-tickets')
df = pd.DataFrame(dataset['train'])

print('Columns:', df.columns.tolist())
print('\nShape:', df.shape)
print('\nSample row:')
print(df.iloc[0])
print('\nMissing values:')
print(df.isnull().sum())
