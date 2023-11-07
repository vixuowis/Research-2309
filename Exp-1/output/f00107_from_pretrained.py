from typing import *
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')
