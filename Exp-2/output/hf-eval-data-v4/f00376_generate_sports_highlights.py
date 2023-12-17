# requirements_file --------------------

!pip install -U transformers numpy torch

# function_import --------------------

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

# function_code --------------------

def generate_sports_highlights(video_frames):
    # The video_frames should be a list of numpy arrays representing video frames
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')

    inputs = processor(video_frames, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    return predicted_class

# test_function_code --------------------

def test_generate_sports_highlights():
    print("Testing started.")
    # Simulated video frames as input
    simulated_video = [np.random.randn(3, 224, 224) for _ in range(16)]

    print("Testing generate_sports_highlights function.")
    predicted_class = generate_sports_highlights(simulated_video)
    assert type(predicted_class) == str, f"Test failed: The predicted class should be a string, but got {type(predicted_class)}"
    print("Testing finished.")

# Run the test
test_generate_sports_highlights()