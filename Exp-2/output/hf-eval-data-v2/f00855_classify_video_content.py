# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def classify_video_content(video):
    """
    Classify a video's content into multiple categories like 'sports', 'comedy', and 'news'.

    Args:
        video (list): A list of frames representing a video. Each frame is a 3D numpy array (height, width, channels).

    Returns:
        str: The category with the highest probability for the given video input.
    """
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2')
    inputs = processor(video, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_video_content():
    """
    Test the function classify_video_content.
    """
    video = list(np.random.randn(16, 3, 224, 224))
    category = classify_video_content(video)
    assert isinstance(category, str), 'The output should be a string representing the category.'

# call_test_function_code --------------------

test_classify_video_content()