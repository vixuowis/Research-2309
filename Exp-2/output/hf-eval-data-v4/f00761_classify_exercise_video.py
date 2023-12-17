# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_exercise_video(video_frames):
    """
    Classify the type of exercise in a given video.

    Parameters:
        video_frames (list): A list of video frames, where each frame is a numpy array of shape (3, 224, 224).

    Returns:
        str: The label of the predicted exercise class.
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

def test_classify_exercise_video():
    print("Testing classify_exercise_video function.")
    video_frames = [np.random.randn(3, 224, 224) for _ in range(8)]  # Simulate 8 video frames

    predicted_label = classify_exercise_video(video_frames)

    assert isinstance(predicted_label, str), f"Expected a string label, got {type(predicted_label)}"
    print("All tests passed!")

# Run the test function
test_classify_exercise_video()