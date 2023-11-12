# function_import --------------------

from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def classify_sports_videos(video):
    '''
    Classify actions of athletes in sports videos.

    Args:
        video (list): A list of frames from a video. Each frame is a 3D numpy array (channels, height, width).

    Returns:
        dict: The outputs of the model, including the loss and the logits.

    Raises:
        ValueError: If the input video is not a list or if it's empty.
    '''
    if not isinstance(video, list) or not video:
        raise ValueError('The input video should be a non-empty list of frames.')

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base-short-ssv2')

    pixel_values = feature_extractor(video, return_tensors='pt').pixel_values
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    num_frames = len(video)
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame

    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

    return outputs

# test_function_code --------------------

def test_classify_sports_videos():
    '''
    Test the function classify_sports_videos.
    '''
    # Test with random data
    num_frames = 16
    video = list(np.random.randn(num_frames, 3, 224, 224))
    outputs = classify_sports_videos(video)
    assert 'loss' in outputs
    assert 'logits' in outputs

    # Test with empty list
    try:
        classify_sports_videos([])
    except ValueError as e:
        assert str(e) == 'The input video should be a non-empty list of frames.'

    # Test with non-list input
    try:
        classify_sports_videos('not a list')
    except ValueError as e:
        assert str(e) == 'The input video should be a non-empty list of frames.'

    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_classify_sports_videos()