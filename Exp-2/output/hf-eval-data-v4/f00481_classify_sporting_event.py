# requirements_file --------------------

!pip install -U transformers torch numpy

# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch
import numpy as np

# function_code --------------------

def classify_sporting_event(video):
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics')
    inputs = processor(video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_sporting_event():
    print('Testing started.')
    sample_video = list(np.random.randn(16, 3, 224, 224))  # Replace with a real video

    # Test case 1
    print('Testing case [1/1] started.')
    result = classify_sporting_event(sample_video)
    assert isinstance(result, str), f'Test case [1/1] failed: The result is not a string but {type(result)}.'
    print('Testing case [1/1] passed.')
    print('Testing finished.')