from typing import *
from transformers import pipeline

# The pipeline can also iterate over an entire dataset for any task you like. For this example, let's choose automatic speech recognition as our task:
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
