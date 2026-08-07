from datasets import load_dataset
import pandas as pd

dataset = load_dataset('Tobi-Bueck/customer-support-tickets')
df = pd.DataFrame(dataset['train'])

df_en = df[df['language'] == 'en'].copy()
print('English tickets:', df_en.shape)

print('Queue distribution:')
print(df_en['queue'].value_counts())

print('Priority distribution:')
print(df_en['priority'].value_counts())

df_en.to_csv('data/tickets_english.csv', index=False)
print('Saved filtered dataset to data/tickets_english.csv')
