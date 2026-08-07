import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/tickets_english.csv')

# Drop rows with missing subject or body
df = df.dropna(subset=['subject', 'body', 'queue'])

# Combine subject + body into one text field
df['text'] = df['subject'].astype(str) + ' ' + df['body'].astype(str)

# Keep only what we need
df_clean = df[['text', 'queue']].copy()

print('Final dataset shape:', df_clean.shape)
print('\nQueue distribution after cleaning:')
print(df_clean['queue'].value_counts())

# Train/test split, stratified by queue
train_df, test_df = train_test_split(
    df_clean, test_size=0.2, random_state=42, stratify=df_clean['queue']
)

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print('\nTrain shape:', train_df.shape)
print('Test shape:', test_df.shape)
print('Saved to data/train.csv and data/test.csv')
