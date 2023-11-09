# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def analyze_workout_video(video):
    """
    Analyze a workout video using the pre-trained model 'MCG-NJU/videomae-base-short'.

    Args:
        video (list): A list of numpy arrays representing the frames of the workout video. Each frame should be resized to 224x224 pixels.

    Returns:
        outputs (torch.Tensor): The output features from the model that can be used for providing customized workout plans.
    """
    num_frames = 16
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base-short')

    pixel_values = processor(images=video, return_tensors='pt').pixel_values
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame

    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

    return outputs

# test_function_code --------------------

def test_analyze_workout_video():
    """
    Test the function analyze_workout_video.
    """
    # Generate a random video for testing
    video = list(np.random.randn(16, 3, 224, 224))
    outputs = analyze_workout_video(video)

    # Check the type of the output
    assert isinstance(outputs, torch.Tensor), 'The output should be a torch.Tensor.'

    # Check the shape of the output
    assert len(outputs.shape) == 2, 'The output should be a 2D tensor.'

# call_test_function_code --------------------

test_analyze_workout_video()