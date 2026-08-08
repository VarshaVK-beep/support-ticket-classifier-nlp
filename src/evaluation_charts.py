import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report

test_df = pd.read_csv('data/test.csv')
test_df = test_df.sample(n=min(300, len(test_df)), random_state=42)

with open('src/label_mappings.json') as f:
    mappings = json.load(f)
label2id = mappings['label2id']
id2label = {int(k): v for k, v in mappings['id2label'].items()}

model = AutoModelForSequenceClassification.from_pretrained('src/final_model')
tokenizer = AutoTokenizer.from_pretrained('src/final_model')
model.eval()

preds = []
true_labels = test_df['queue'].map(label2id).tolist()

with torch.no_grad():
    for text in test_df['text']:
        inputs = tokenizer(text, truncation=True, max_length=256, padding='max_length', return_tensors='pt')
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

labels_sorted = sorted(id2label.keys())
label_names = [id2label[i] for i in labels_sorted]

# Confusion matrix heatmap
cm = confusion_matrix(true_labels, preds, labels=labels_sorted)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Ticket Classification')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('notebooks/plots_confusion_matrix.png', dpi=150)
plt.close()

# Per-class F1 bar chart
report = classification_report(true_labels, preds, target_names=label_names, output_dict=True, zero_division=0)
f1_scores = [report[name]['f1-score'] for name in label_names]

plt.figure(figsize=(10, 6))
colors = sns.color_palette('viridis', len(label_names))
bars = plt.barh(label_names, f1_scores, color=colors)
plt.xlabel('F1-Score')
plt.title('Per-Class F1-Score')
plt.xlim(0, 1)
for i, v in enumerate(f1_scores):
    plt.text(v + 0.02, i, f'{v:.2f}', va='center')
plt.tight_layout()
plt.savefig('notebooks/plots_per_class_f1.png', dpi=150)
plt.close()

print('Saved: notebooks/plots_confusion_matrix.png')
print('Saved: notebooks/plots_per_class_f1.png')
