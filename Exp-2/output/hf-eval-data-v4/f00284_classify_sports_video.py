# requirements_file --------------------

!pip install -U transformers numpy torch

# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_video(video_frames):
    """
    Classify the given sports video frames using the pre-trained TimeSformer model.

    Args:
        video_frames (List[np.ndarray]): A list of numpy arrays representing the video frames.

    Returns:
        str: The predicted class name.
    """
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-hr-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-hr-finetuned-k600')

    inputs = processor(images=video_frames, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_name = model.config.id2label[predicted_class_idx]
    return predicted_class_name

# test_function_code --------------------

def test_classify_sports_video():
    print("Testing classify_sports_video function.")
    # Assuming `sample_video_frames` is a pre-loaded list of frames from a sports video.
    sample_video_frames = list(np.random.randn(16, 3, 448, 448))
    predicted_class = classify_sports_video(sample_video_frames)
    assert type(predicted_class) == str, f"Expected prediction to be a string, got {type(predicted_class)}"
    print("Test passed - the function 'classify_sports_video' correctly returns a string as the predicted class name.")

# Running the test function
print("Running tests...")
test_classify_sports_video()