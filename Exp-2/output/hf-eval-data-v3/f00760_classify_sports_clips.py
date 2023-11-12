# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_clips(video):
    """
    Classify the type of sports being played in the video using a pre-trained model.

    Args:
        video (list): A list of numpy arrays representing frames of the video.

    Returns:
        str: The predicted class of the sports being played in the video.
    """
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k400')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k400')
    inputs = processor(video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_sports_clips():
    """
    Test the function classify_sports_clips.
    """
    video = list(np.random.randn(8, 3, 224, 224))
    predicted_class = classify_sports_clips(video)
    assert isinstance(predicted_class, str), 'The output should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_classify_sports_clips()