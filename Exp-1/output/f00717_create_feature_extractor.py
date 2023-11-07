from typing import *
from transformers import Wav2Vec2FeatureExtractor

def create_feature_extractor():
    """
    Create a feature extractor associated with the model you're using.

    Returns:
        feature_extractor (Wav2Vec2FeatureExtractor): The feature extractor object.
    """
    feature_extractor = Wav2Vec2FeatureExtractor()
    return feature_extractor
