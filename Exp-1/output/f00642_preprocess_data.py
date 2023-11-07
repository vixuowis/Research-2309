from typing import *
from transformers import SpeechT5Processor

def preprocess_data(checkpoint):
    processor = SpeechT5Processor.from_pretrained(checkpoint)
    return processor
