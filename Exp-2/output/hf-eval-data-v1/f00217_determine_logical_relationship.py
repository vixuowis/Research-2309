import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Function to determine the logical relationship between two given sentences
# Uses the Hugging Face Transformers library and a pretrained model for sequence classification
# The model checkpoint is 'cointegrated/rubert-base-cased-nli-threeway'
def determine_logical_relationship(text1, text2):
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    with torch.inference_mode():
        out = model(**tokenizer(text1, text2, return_tensors='pt').to(model.device))
        proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
    result = {v: proba[k] for k, v in model.config.id2label.items()}
    return result