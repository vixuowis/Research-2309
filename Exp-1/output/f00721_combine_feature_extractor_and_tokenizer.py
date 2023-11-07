from typing import *
from transformers import Wav2Vec2Processor

def combine_feature_extractor_and_tokenizer(feature_extractor, tokenizer):
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor
