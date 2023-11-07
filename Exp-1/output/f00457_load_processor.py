from typing import *
from transformers import AutoProcessor

def load_processor():
    # Load a Wav2Vec2 processor to process the audio signal
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
