from typing import *
from transformers import pipeline

def summarizer(text: str) -> List[Dict[str, str]]:
    summarizer = pipeline(task='summarization')
    return summarizer(text)
