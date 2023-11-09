# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_sports_video(video):
    """
    Classify the given sports video using the pre-trained model 'MCG-NJU/videomae-large-finetuned-kinetics'.

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
    """
    Test the function classify_sports_video.
    """
    video = list(np.random.randn(16, 3, 224, 224))
    predicted_class = classify_sports_video(video)
    assert isinstance(predicted_class, str)

# call_test_function_code --------------------

test_classify_sports_video()