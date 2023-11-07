from typing import *
from transformers import SpeechT5ForTextToSpeech

def run_inference_manually():
    # Load the model from the 🤗 Hub:
    model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
