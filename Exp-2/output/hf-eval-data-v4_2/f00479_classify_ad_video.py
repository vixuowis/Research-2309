# requirements_file --------------------

!pip install -U transformers numpy torch

# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_ad_video(video_frames):
    """
    Classify an advertisement video into categories based on pre-trained model.

    Args:
        video_frames (list): A list of 3D numpy arrays representing the video frames.

    Returns:
        str: Predicted category of the advertisement video.

    Raises:
        ValueError: If video_frames is not a list of 3D numpy arrays.
    """
    # Validate the input format
    if not isinstance(video_frames, list) or not all(isinstance(frame, np.ndarray) and frame.ndim == 3 for frame in video_frames):
        raise ValueError('Input must be a list of 3D numpy arrays')

    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k600')

    # Process the video frames
    inputs = processor(images=video_frames, return_tensors='pt')

    # Pass the processed inputs through the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class index
    predicted_class_idx = logits.argmax(-1).item()

    # Return the predicted class
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_ad_video():
    print("Testing started.")
    # Use a random set of frames to simulate a video
    video = [np.random.randn(3, 224, 224) for _ in range(8)]

    # Test case 1: Correct input format
    print("Testing case [1/1] started.")
    predicted_label = classify_ad_video(video)
    assert isinstance(predicted_label, str), f"Test case [1/1] failed: Expected a string label, got {type(predicted_label).__name__}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_ad_video()