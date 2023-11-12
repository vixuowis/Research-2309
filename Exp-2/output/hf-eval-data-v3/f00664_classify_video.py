# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_video(video_frames):
    """
    Classify the content of a video lecture automatically using Timesformer model.

    Args:
        video_frames (list): A list of video frames.

    Returns:
        str: The predicted class of the video content.
    """
    processor = AutoImageProcessor.from_pretrained('fcakyon/timesformer-hr-finetuned-k400')
    model = TimesformerForVideoClassification.from_pretrained('fcakyon/timesformer-hr-finetuned-k400')
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
    video_frames = [np.random.randn(3, 448, 448) for _ in range(16)]
    predicted_class = classify_video(video_frames)
    assert isinstance(predicted_class, str), 'The result should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_classify_video()