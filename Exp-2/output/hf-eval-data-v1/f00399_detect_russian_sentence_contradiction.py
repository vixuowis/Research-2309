import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Function to detect if one Russian sentence logically contradicts the information provided by another Russian sentence
# Uses the 'cointegrated/rubert-base-cased-nli-threeway' model from Hugging Face Transformers

def detect_russian_sentence_contradiction(sentence1: str, sentence2: str) -> bool:
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    with torch.inference_mode():
        out = model(**tokenizer(sentence1, sentence2, return_tensors='pt').to(model.device))
        proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
    predicted_label = {v: proba[k] for k, v in model.config.id2label.items()}
    return predicted_label['contradiction'] > predicted_label['neutral'] and predicted_label['contradiction'] > predicted_label['entailment']