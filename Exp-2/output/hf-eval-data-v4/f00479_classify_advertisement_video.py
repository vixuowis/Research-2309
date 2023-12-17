# requirements_file --------------------

!pip install -U transformers torch numpy

# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_advertisement_video(video_frames):
    """
    Classify an advertisement video into categories based on the pre-trained
    'facebook/timesformer-base-finetuned-k600' model.

    :param video_frames: List of 3D numpy arrays representing the video frames
    :return: Predicted class label of the video
    """
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k600')
    inputs = processor(images=video_frames, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_advertisement_video():
    print("Testing classify_advertisement_video function.")
    # Assuming a dummy video frame array simulating random 8 frames of a video
    dummy_video_frames = [np.random.randn(3, 224, 224) for _ in range(8)]

    # Testing with dummy video frames
    predicted_class = classify_advertisement_video(dummy_video_frames)
    assert isinstance(predicted_class, str), f"Expected string class label, got {type(predicted_class)}"
    print("Test passed!")