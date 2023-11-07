from typing import *
from transformers import AutoFeatureExtractor

def load_feature_extractor():
	# Load a Wav2Vec2 feature extractor to process the audio signal
	feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
