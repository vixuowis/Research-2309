from typing import *
import torch

def get_transcription(logits, processor):
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = processor.batch_decode(predicted_ids)
	return transcription
