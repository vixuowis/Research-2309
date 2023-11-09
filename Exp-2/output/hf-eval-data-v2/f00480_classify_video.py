# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_video(video_path):
    """
    Classify the actions or activities occurring in a video using a pre-trained model.

    Args:
        video_path (str): The path to the video file to be classified.

    Returns:
        str: The predicted class of action or activity occurring in the video.
    """
    video_frames = load_video_frames(video_path)
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-hr-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-hr-finetuned-k600')
    inputs = processor(images=video_frames, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_video():
    """
    Test the classify_video function.
    """
    video_path = 'path_to_test_video'
    predicted_class = classify_video(video_path)
    assert isinstance(predicted_class, str)

# call_test_function_code --------------------

test_classify_video()