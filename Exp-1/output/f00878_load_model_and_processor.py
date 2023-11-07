from typing import *
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

def load_model_and_processor(model_id):
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    return processor, model
