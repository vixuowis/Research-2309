from typing import *
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

def classify_language(audio_path):
	"""
	Classify the language of an audio file.

	Args:
		audio_path (str): The path to the audio file.

	Returns:
		str: The predicted language.
	"""
	audio, _ = torchaudio.load(audio_path)

	processor = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
	input_features = processor(audio, sampling_rate=16000, return_tensors='pt').input_values

	model = Wav2Vec2ForSequenceClassification.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')

	with torch.no_grad():
		outputs = model(input_features)
		predictions = torch.argmax(outputs.logits, dim=1)

	return processor.decode(predictions)
