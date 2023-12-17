# requirements_file --------------------

!pip install -U transformers torch numpy

# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_video_content(video):
    """
    Classify the content of a given video into multiple categories (sports, comedy, news, etc.).

    Parameters:
    video (union[list, np.ndarray, torch.Tensor]): The video data consisting of frames.

    Returns:
    str: The category with the highest probability.
    """
    # Load the pre-trained model and processor
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2')

    # Preprocess the video frames
    inputs = processor(video, return_tensors='pt')

    # Predict the category of the video
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Return the predicted category
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_video_content():
    print("Testing started.")
    # Prepare a random video clip for testing
    video_clip = [np.random.randn(3, 224, 224) for _ in range(16)]

    # Test case 1: Check if the function returns a string
    print("Testing case [1/1] started.")
    result = classify_video_content(video_clip)
    assert isinstance(result, str), f"Test case [1/1] failed: Result should be a string, got {type(result)}"
    print("Testing finished.")