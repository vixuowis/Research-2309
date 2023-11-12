# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def classify_sports_video(video):
    '''
    Classify the sports content in a video using the pre-trained model 'MCG-NJU/videomae-base'.

    Args:
        video (list): A list of frames in the video. Each frame is a 3D numpy array (channels, height, width).

    Returns:
        torch.Tensor: The output from the model, which includes the loss and the logits.
    '''
    num_frames = 16
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base')

    pixel_values = processor(video, return_tensors='pt').pixel_values
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame

    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    return outputs

# test_function_code --------------------

def test_classify_sports_video():
    '''
    Test the function classify_sports_video.
    '''
    # Test case: a video with random pixel values
    video = list(np.random.randn(16, 3, 224, 224))
    outputs = classify_sports_video(video)
    assert isinstance(outputs, torch.Tensor), 'The output should be a torch.Tensor.'

    # Test case: a video with all zero pixel values
    video = list(np.zeros((16, 3, 224, 224)))
    outputs = classify_sports_video(video)
    assert isinstance(outputs, torch.Tensor), 'The output should be a torch.Tensor.'

    print('All tests passed.')

# call_test_function_code --------------------

test_classify_sports_video()