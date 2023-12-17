# requirements_file --------------------

!pip install -U transformers numpy torch

# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_exercise_video(video_frames):
    """
    Classifies an exercise video into categories based on pre-trained model.

    Args:
        video_frames (list): A list of video frames, where each frame is a numpy array.

    Returns:
        str: Predicted category label for the input video.

    Raises:
        ValueError: If video_frames is not a list or it's empty.
    """
    if not isinstance(video_frames, list) or not video_frames:
        raise ValueError('video_frames must be a non-empty list.')

    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k600')

    inputs = processor(images=video_frames, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_exercise_video():
    print('Testing started.')
    sample_video = [np.random.randn(3, 224, 224) for _ in range(8)]

    # Test with a proper video sample
    print('Testing case [1/2] started.')
    predicted_label = classify_exercise_video(sample_video)
    assert isinstance(predicted_label, str), f'Test case [1/2] failed: Expected a string label, got {type(predicted_label)}.'

    # Test with invalid input (empty list)
    try:
        print('Testing case [2/2] started.')
        classify_exercise_video([])
        assert False, 'Test case [2/2] failed: ValueError not raised for empty list.'
    except ValueError as e:
        assert str(e) == 'video_frames must be a non-empty list.', f'Test case [2/2] failed: {e}'
    print('Testing finished.')

# call_test_function_line --------------------

test_classify_exercise_video()