# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_video(video):
    """
    Classify sports video using TimesformerForVideoClassification model from Hugging Face Transformers.

    Args:
        video (list): A list of numpy arrays representing the video frames. Each frame should have dimensions 3x448x448.

    Returns:
        str: The predicted class for the sports video.
    """
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-hr-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-hr-finetuned-k600')

    inputs = processor(images=video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_sports_video():
    """
    Test the classify_sports_video function.
    """
    # Test case: Random video frames
    video = list(np.random.randn(16, 3, 448, 448))
    predicted_class = classify_sports_video(video)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'

    # Test case: All zero frames
    video = list(np.zeros((16, 3, 448, 448)))
    predicted_class = classify_sports_video(video)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'

    # Test case: All one frames
    video = list(np.ones((16, 3, 448, 448)))
    predicted_class = classify_sports_video(video)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_sports_video()