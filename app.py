import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Support Ticket Classifier")
st.write("Paste a customer support ticket to predict which department it should be routed to.")

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("src/final_model")
    tokenizer = AutoTokenizer.from_pretrained("src/final_model")
    with open("src/label_mappings.json") as f:
        mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings["id2label"].items()}
    return model, tokenizer, id2label

model, tokenizer, id2label = load_model()

text = st.text_area("Ticket text", placeholder="e.g. My internet has been down since yesterday and I can't reach support...")

if st.button("Classify Ticket"):
    if not text.strip():
        st.warning("Please enter some ticket text.")
    else:
        inputs = tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()

        st.success(f"Predicted queue: **{id2label[pred_id]}**")
        st.write(f"Confidence: {confidence:.1%}")