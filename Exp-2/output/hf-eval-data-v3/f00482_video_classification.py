# function_import --------------------

from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def video_classification(video):
    '''
    Classify the input video using pretrained VideoMAE model from Hugging Face.

    Args:
        video (list): A list of image frames representing the video.

    Returns:
        outputs (torch.Tensor): The output tensor from the model, representing the video classification.
    '''
    # Load the pretrained VideoMAE model and feature extractor
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base-short-ssv2')

    # Convert video input into pixel values
    pixel_values = feature_extractor(video, return_tensors='pt').pixel_values

    # Pass the pixel values into the model to obtain predictions
    outputs = model(pixel_values)

    return outputs

# test_function_code --------------------

def test_video_classification():
    '''
    Test the video_classification function with a random video.
    '''
    # Generate a random video
    video = list(np.random.randn(16, 3, 224, 224))

    # Classify the video
    outputs = video_classification(video)

    # Check the output type and shape
    assert isinstance(outputs, torch.Tensor), 'Output should be a torch.Tensor'
    assert outputs.shape == (1, 16, 3, 224, 224), 'Output shape should be (1, 16, 3, 224, 224)'

    print('All tests passed.')

# call_test_function_code --------------------

test_video_classification()