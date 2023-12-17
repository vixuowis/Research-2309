# requirements_file --------------------

!pip install -U transformers numpy torch

# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_video(video_frames):
    """
    Classifies the type of sport in the given video frames.

    Args:
        video_frames (list): A list of video frames, where each frame is a numpy array of shape (3, 224, 224).

    Returns:
        str: The predicted category of the sport from the video frames.
    """
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics')

    inputs = processor(video_frames, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
    return predicted_class

# test_function_code --------------------

def test_classify_sports_video():
    print("Testing started.")
    sample_video = list(np.random.randn(16, 3, 224, 224))  # Generate random video frames as sample

    # Test case 1: Check if the function returns a string
    print("Testing case [1/1] started.")
    result = classify_sports_video(sample_video)
    assert isinstance(result, str), f"Test case [1/1] failed: Result should be a string, got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_sports_video()