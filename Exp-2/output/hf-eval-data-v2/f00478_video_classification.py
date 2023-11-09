# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def video_classification(video):
    '''
    This function classifies the activities in a video using the pretrained model 'MCG-NJU/videomae-base'.

    Args:
        video (list): A list of frames in the video. Each frame is a 3D numpy array with shape (3, 224, 224).

    Returns:
        loss (torch.Tensor): The loss calculated by the model for the given video clip.
    '''
    num_frames = 16
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base')
    pixel_values = processor(video, return_tensors='pt').pixel_values
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    return outputs.loss

# test_function_code --------------------

def test_video_classification():
    '''
    This function tests the video_classification function by generating a random video and checking the type and shape of the output.
    '''
    video = list(np.random.randn(16, 3, 224, 224))
    loss = video_classification(video)
    assert isinstance(loss, torch.Tensor), 'The output should be a torch.Tensor.'
    assert loss.shape == (), 'The output tensor should be a scalar.'

# call_test_function_code --------------------

test_video_classification()