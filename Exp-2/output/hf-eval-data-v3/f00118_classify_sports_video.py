# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_video(video):
    """
    Classify the sports in the given video using a pre-trained model.

    Args:
        video (list): A list of video frames. Each frame is a 3D numpy array with shape (3, 224, 224).

    Returns:
        str: The predicted sports category.
    """
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics')

    inputs = processor(video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_sports_video():
    """Test the classify_sports_video function."""
    video = list(np.random.randn(16, 3, 224, 224))
    result = classify_sports_video(video)
    assert isinstance(result, str), 'The result should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_classify_sports_video()