import pandas as pd
from transformers import AutoTokenizer

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Encode labels
labels = sorted(train_df['queue'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

train_df['label'] = train_df['queue'].map(label2id)
test_df['label'] = test_df['queue'].map(label2id)

print('Label mapping:')
print(label2id)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Test tokenization on one example
sample_text = train_df['text'].iloc[0]
tokens = tokenizer(sample_text, truncation=True, max_length=256, padding='max_length')
print('\nSample tokenized length:', len(tokens['input_ids']))
print('Sample text preview:', sample_text[:100])

# Save label mappings for later use
import json
with open('src/label_mappings.json', 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f)

print('\nSaved label mappings to src/label_mappings.json')
