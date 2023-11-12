# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_video(video_frames):
    """
    Classify the actions or activities occurring in the video using the pre-trained model 'facebook/timesformer-hr-finetuned-k600'.

    Args:
        video_frames (list): A list of video frames.

    Returns:
        str: The predicted class of action or activity occurring in the video.
    """
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
    Test the function classify_video.
    """
    video_frames = [np.random.randn(3, 448, 448) for _ in range(16)]
    predicted_class = classify_video(video_frames)
    assert isinstance(predicted_class, str), 'The result should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_classify_video()