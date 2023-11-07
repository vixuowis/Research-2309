from typing import *
from model import Model

def generate_speech(input_ids, speaker_embeddings):
    	model = Model()
    	return model.generate_speech(input_ids, speaker_embeddings)
