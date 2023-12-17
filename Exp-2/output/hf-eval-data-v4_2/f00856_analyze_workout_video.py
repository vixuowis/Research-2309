# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import torch

# function_code --------------------

def analyze_workout_video(video_frames, num_frames=16):
    """
    Analyses workout video frames for offering customized workout plans.

    Args:
        video_frames (List[np.ndarray]): A list of numpy arrays representing frames of a workout video.
        num_frames (int): The number of frames to analyze. Defaults to 16.

    Returns:
        torch.Tensor: The output features from the VideoMAE model.

    Raises:
        ValueError: If less than num_frames are provided.

    """
    if len(video_frames) < num_frames:
        raise ValueError(f'Expected at least {num_frames} frames, but got {len(video_frames)}.')

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base-short')

    pixel_values = processor(images=video_frames, return_tensors='pt').pixel_values
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame

    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

    return outputs.last_hidden_state

# test_function_code --------------------

def test_analyze_workout_video():
    print("Testing started.")
    # Assuming load_dataset() and a hypothetical dataset is available
    dataset = load_dataset("workout_video_dataset")
    sample_data = dataset[0]  # Get a sample from the dataset

    # Test case 1: Process a valid video with 16 frames
    print("Testing case [1/3] started.")
    result = analyze_workout_video(sample_data['frames'], num_frames=16)
    assert result is not None, f"Test case [1/3] failed: result should not be None"

    # Test case 2: Process video with less than num_frames
    print("Testing case [2/3] started.")
    try:
        analyze_workout_video(sample_data['frames'][:10], num_frames=16)
        assert False, "Test case [2/3] failed: ValueError was not raised"
    except ValueError as e:
        assert '16 frames' in str(e), f"Test case [2/3] failed: {e}"

    # Test case 3: Process video with exactly num_frames
    print("Testing case [3/3] started.")
    result = analyze_workout_video(sample_data['frames'][:16], num_frames=16)
    assert result.shape[0] == 16, f"Test case [3/3] failed: Incorrect number of frames processed"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_workout_video()