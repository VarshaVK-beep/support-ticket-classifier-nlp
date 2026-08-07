import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

app = FastAPI(title='Support Ticket Classifier API')

model = AutoModelForSequenceClassification.from_pretrained('src/final_model')
tokenizer = AutoTokenizer.from_pretrained('src/final_model')

with open('src/label_mappings.json') as f:
    mappings = json.load(f)
id2label = {int(k): v for k, v in mappings['id2label'].items()}

class Ticket(BaseModel):
    text: str

@app.get('/')
def root():
    return {'status': 'Support Ticket Classifier API is running'}

@app.post('/classify')
def classify(ticket: Ticket):
    inputs = tokenizer(ticket.text, truncation=True, max_length=256, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()
    return {
        'predicted_queue': id2label[pred_id],
        'confidence': round(confidence, 3)
    }
