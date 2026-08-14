import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Support Ticket Classifier")
st.write("Paste a customer support ticket to predict which department it should be routed to.")
st.caption("⚠️ Trained on telecom/software support tickets — accuracy ~42% on a 10-class problem. Best results with clear billing, connectivity, or outage-related issues.")

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("Varshavk12/support-ticket-classifier")
    tokenizer = AutoTokenizer.from_pretrained("Varshavk12/support-ticket-classifier")
    with open("src/label_mappings.json") as f:
        mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings["id2label"].items()}
    return model, tokenizer, id2label

model, tokenizer, id2label = load_model()

# Keyword rules — quick, interpretable signal alongside the model
KEYWORD_RULES = {
    "Billing and Payments": ["charged", "refund", "invoice", "payment", "billing", "overcharged"],
    "Returns and Exchanges": ["return", "exchange", "damaged", "defective", "wrong item"],
    "Service Outages and Maintenance": ["outage", "down", "not working", "service down", "maintenance"],
    "Sales and Pre-Sales": ["pricing", "upgrade", "plan", "purchase", "quote", "interested in buying"],
    "Human Resources": ["hr", "benefits", "payroll", "leave", "employee"],
    "IT Support": ["password", "login", "vpn", "access", "account locked"],
    "Technical Support": ["error", "bug", "crash", "not loading", "broken"],
}

def keyword_match(text):
    text_lower = text.lower()
    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                return category, kw
    return None, None

text = st.text_area("Ticket text", placeholder="e.g. My internet has been down since yesterday and I can't reach support...")

if st.button("Classify Ticket"):
    if not text.strip():
        st.warning("Please enter some ticket text.")
    else:
        # Model prediction
        inputs = tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()

        st.success(f"Model prediction: **{id2label[pred_id]}**")
        st.write(f"Confidence: {confidence:.1%}")

        # Keyword rule check
        kw_category, matched_word = keyword_match(text)
        if kw_category:
            st.info(f"🔑 Keyword match: **{kw_category}** (matched word: \"{matched_word}\")")
        else:
            st.caption("No strong keyword match found — relying on model prediction only.")