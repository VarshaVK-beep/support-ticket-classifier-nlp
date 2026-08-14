## Results

Fine-tuned DistilBERT on a 28,000+ row multilingual support ticket dataset (filtered to English), classifying tickets into a 10-class label set.

- Accuracy: 42.3%
- Weighted F1: 0.38
- Majority-class baseline: ~30% accuracy

Per-class precision/recall/F1 breakdown generated via `src/evaluate_model.py` to diagnose weak-performing classes on the imbalanced label set.

## Quickstart

```bash
git clone https://github.com/VarshaVK-beep/support-ticket-classifier-nlp.git
cd support-ticket-classifier-nlp
pip install -r requirements.txt

python src/prepare_data.py
python src/tokenize_setup.py
python src/finetune.py
python src/evaluate_model.py

uvicorn api.main:app --reload
```

Visit http://localhost:8000/docs for the interactive Swagger UI.