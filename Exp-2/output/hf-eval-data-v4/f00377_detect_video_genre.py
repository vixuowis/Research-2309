# requirements_file --------------------

!pip install -U transformers decord torch numpy

# function_import --------------------

from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
from decord import VideoReader
import torch
import numpy as np

# function_code --------------------

def detect_video_genre(video_path):
    """
    Detect the genre of a movie by classifying the actions in a given video.

    Parameters:
        video_path (str): The file path to the video file.

    Returns:
        str: The predicted genre of the movie.
    """
    videoreader = VideoReader(video_path)
    model = VideoMAEForVideoClassification.from_pretrained('nateraw/videomae-base-finetuned-ucf101')
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('nateraw/videomae-base-finetuned-ucf101')

    frames = videoreader.get_batch(list(range(0, len(videoreader), 4)))  # Sample every 4 frames
    inputs = feature_extractor(list(frames.asnumpy()), return_tensors='pt')
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_detect_video_genre():
    print("Testing started.")
    sample_video = 'sample_video.mp4'  # Replace with a valid video file path

    # Test case 1: Detect genre of a sample video
    print("Testing case [1/1] started.")
    genre = detect_video_genre(sample_video)
    assert isinstance(genre, str), f"Test case [1/1] failed: Expected a string, got {type(genre)}"
    print("Testing finished.")