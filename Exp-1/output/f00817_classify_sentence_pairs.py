from typing import *
from transformers import pipeline

def classify_sentence_pairs(sentences):
    classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
    results = classifier(sentences)
    return results
