# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_video(video):
    """
    Classify the exercise in a video using the TimesformerForVideoClassification model.

    Args:
        video (list): A list of numpy arrays representing the video frames.

    Returns:
        str: The predicted exercise label.

    Raises:
        ValueError: If the video frames are not in the correct format.
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

def test_classify_video():
    """Test the classify_video function."""
    video = list(np.random.randn(8, 3, 224, 224))
    result = classify_video(video)
    assert isinstance(result, str), 'The result should be a string.'
    assert result in model.config.id2label.values(), 'The result should be a valid label.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_video()