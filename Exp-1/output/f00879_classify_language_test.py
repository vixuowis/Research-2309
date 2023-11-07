from f00879_classify_language import *
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import pytest

@pytest.fixture

def test_classify_language():
	# Test case 1
	audio_path = 'path/to/audio1.wav'
	expected_language = 'English'
	assert classify_language(audio_path) == expected_language

	# Test case 2
	audio_path = 'path/to/audio2.wav'
	expected_language = 'French'
	assert classify_language(audio_path) == expected_language

	# Test case 3
	audio_path = 'path/to/audio3.wav'
	expected_language = 'Spanish'
	assert classify_language(audio_path) == expected_language
