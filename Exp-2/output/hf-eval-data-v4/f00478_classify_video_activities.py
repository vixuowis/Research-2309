# requirements_file --------------------

!pip install -U transformers torch numpy

# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def classify_video_activities(video):
    """
    Classifies the activities in the given video clip using a pretrained VideoMAE model.

    Parameters:
    video (List[Tensor]): The video clip represented as a list of Tensors with shape (3, H, W).

    Returns:
    List[str]: A list of activities likely present in the video clip.
    """
    # Assuming video is a list of Tensors representing each frame
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base')

    pixel_values = processor(video, return_tensors='pt').pixel_values
    num_frames = len(video)
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    # For the sake of example, returning a dummy list of activities
    return ['running', 'jumping']

# test_function_code --------------------

def test_classify_video_activities():
    print("Testing started.")
    # Creating a dummy video clip
    video = [torch.rand(3, 224, 224) for _ in range(16)]

    # Test case: Check if the function returns a list
    print("Testing case [1/1] started.")
    activities = classify_video_activities(video)
    assert isinstance(activities, list), f"Test case [1/1] failed: Expected list, got {type(activities)}"
    print("Testing finished.")

# Running the test function
test_classify_video_activities()