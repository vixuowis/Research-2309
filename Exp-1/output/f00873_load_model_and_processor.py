from typing import *
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

def load_model_and_processor(model_id):
	processor = AutoProcessor.from_pretrained(model_id)
	model = Wav2Vec2ForCTC.from_pretrained(model_id)
	return processor, model
