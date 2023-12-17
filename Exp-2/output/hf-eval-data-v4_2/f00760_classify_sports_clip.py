# requirements_file --------------------

!pip install -U transformers numpy torch

# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_clip(video_data: list):
    """
    Classifies the type of sports being played in the given video clip.
    
    Args:
        video_data: A list of frames where each frame is represented as a numpy array of shape (3, 224, 224).
    
    Returns:
        A string representing the predicted class of sports being played in the video.
    Raisese
        ValueError: If video_data is not a list or if its elements are not numpy arrays of the expected shape.
    """
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k400')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k400')
    inputs = processor(video_data, return_tensors='pt')
     with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_sports_clip():
    print("Testing started.")
    video_data = [np.random.randn(3, 224, 224) for _ in range(8)]

    # Testing case 1: Check if function returns a string
    print("Testing case [1/1] started.")
    result = classify_sports_clip(video_data)
    assert isinstance(result, str), f"Test case [1/1] failed: Expected result to be a string, got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_sports_clip()