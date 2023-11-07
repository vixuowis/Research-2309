from typing import *
from transformers import AutoFeatureExtractor

def load_feature_extractor(model_name: str) -> AutoFeatureExtractor:
    '''
    Load a feature extractor to preprocess the audio file and return the `input` as PyTorch tensors

    :param model_name: The name of the pre-trained feature extractor model
    :return: The loaded feature extractor
    '''
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
