from typing import *
from transformers import pipeline

def classify_french_text(text):
    model = 'nlptown/bert-base-multilingual-uncased-sentiment'
    tokenizer = 'nlptown/bert-base-multilingual-uncased-sentiment'
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    result = classifier(text)
    return result
