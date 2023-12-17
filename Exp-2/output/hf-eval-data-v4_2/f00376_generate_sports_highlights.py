# requirements_file --------------------

!pip install -U transformers numpy torch

# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def generate_sports_highlights(video_frames: list) -> str:
    """
    Generate sports highlights by classifying the type of sports activity in the video frames.

    Args:
        video_frames (list): A list of numpy arrays where each array represents a frame in the video sequence.

    Returns:
        str: The predicted sports activity category.

    Raises:
        ValueError: If the `video_frames` is not a list of numpy arrays.
    """
    if not isinstance(video_frames, list) or not all(isinstance(frame, np.ndarray) for frame in video_frames):
        raise ValueError('The video_frames must be a list of numpy arrays.')

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')

    inputs = processor(video_frames, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_category = model.config.id2label[predicted_class_idx]
    return predicted_category

# test_function_code --------------------

def test_generate_sports_highlights():
    print("Testing started.")
    sample_video = list(np.random.randn(16, 3, 224, 224))  # Assuming 16 frames of dummy data

    # Testing case 1: Check type of the function's return value
    print("Testing case [1/3] started.")
    predicted_category = generate_sports_highlights(sample_video)
    assert isinstance(predicted_category, str), f"Test case [1/3] failed: expected str, got {type(predicted_category)}"

    # Testing case 2: Check handling of invalid input
    print("Testing case [2/3] started.")
    try:
        generate_sports_highlights(None)
    except ValueError as e:
        assert str(e) == 'The video_frames must be a list of numpy arrays.', f"Test case [2/3] failed: {e}"

    # Testing case 3: Check for the non-empty result
    print("Testing case [3/3] started.")
    assert predicted_category, "Test case [3/3] failed: the function returned an empty result"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_sports_highlights()