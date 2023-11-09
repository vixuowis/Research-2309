# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_advertisement_video(video):
    """
    Classify the category of an advertisement video using a pre-trained model.

    Args:
        video (list): A list of 3D numpy arrays (channel, height, width) representing video frames.

    Returns:
        str: The predicted class of the advertisement video.
    """
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k600')
    inputs = processor(images=video, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_advertisement_video():
    """
    Test the classify_advertisement_video function with a random video.
    """
    video = list(np.random.randn(8, 3, 224, 224))
    predicted_class = classify_advertisement_video(video)
    assert isinstance(predicted_class, str)

# call_test_function_code --------------------

test_classify_advertisement_video()