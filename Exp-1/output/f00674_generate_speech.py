from typing import *
import torch
from IPython.display import Audio

def generate_speech(vocoder, spectrogram):
    with torch.no_grad():
        speech = vocoder(spectrogram)
    return speech
