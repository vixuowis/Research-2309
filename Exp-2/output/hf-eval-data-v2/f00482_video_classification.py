# function_import --------------------

from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def video_classification(video):
    """
    This function classifies a video using the pretrained VideoMAE model from Hugging Face model hub.

    Args:
        video (list): A list of image frames representing a video.

    Returns:
        outputs (torch.Tensor): The output tensor from the model, representing the video classification.
    """
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    pixel_values = feature_extractor(video, return_tensors='pt').pixel_values
    outputs = model(pixel_values)
    return outputs

# test_function_code --------------------

def test_video_classification():
    """
    This function tests the video_classification function by generating a random video and checking the output type.
    """
    video = list(np.random.randn(16, 3, 224, 224))
    outputs = video_classification(video)
    assert isinstance(outputs, torch.Tensor), 'Output should be a torch.Tensor'

# call_test_function_code --------------------

test_video_classification()