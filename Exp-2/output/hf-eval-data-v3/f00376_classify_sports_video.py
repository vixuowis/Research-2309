# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_video(video):
    '''
    Classify the sports activity happening in the video.

    Args:
        video (list): A list of numpy arrays representing video frames.

    Returns:
        str: The predicted class of the sports activity.
    '''
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')

    inputs = processor(video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_sports_video():
    '''
    Test the function classify_sports_video.
    '''
    video = list(np.random.randn(16, 3, 224, 224))
    predicted_class = classify_sports_video(video)
    assert isinstance(predicted_class, str), 'The output should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_classify_sports_video()