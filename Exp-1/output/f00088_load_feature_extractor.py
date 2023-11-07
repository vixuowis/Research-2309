from typing import *
from transformers import AutoFeatureExtractor

def load_feature_extractor(model_name: str) -> AutoFeatureExtractor:
    """Load a feature extractor to normalize and pad the input.

    Args:
        model_name (str): The name of the pretrained model to use for feature extraction.

    Returns:
        AutoFeatureExtractor: The loaded feature extractor.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return feature_extractor
