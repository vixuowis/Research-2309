# requirements_file --------------------

import subprocess

requirements = ["transformers", "decord"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
from decord import VideoReader
import torch
import numpy as np

# function_code --------------------

def detect_movie_genre(video_path):
    """Detect the genre of a movie based on video actions using a pre-trained model.

    Args:
        video_path (str): The file path to the movie video file.

    Returns:
        str: The predicted genre of the movie based on actions observed in the video.

    Raises:
        FileNotFoundError: If the video file is not found at the specified path.
        RuntimeError: If there is an error during the processing of the video or model prediction.
    """
    # Ensure the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Read the video file
    videoreader = VideoReader(video_path)

    # Load the pre-trained model and feature extractor
    model = VideoMAEForVideoClassification.from_pretrained('nateraw/videomae-base-finetuned-ucf101')
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('nateraw/videomae-base-finetuned-ucf101')

    # Prepare video frames
    frames = videoreader.get_batch(list(range(0, len(videoreader), 4))) # Sample every 4 frames
    inputs = feature_extractor(list(frames.asnumpy()), return_tensors='pt')

    # Predict genre
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax(-1).item()
    genre = model.config.id2label[predicted_label]

    return genre

# test_function_code --------------------

def test_detect_movie_genre():
    print("Testing started.")
    video_path = 'example_video.mp4'  # Replace with a video path obtained from a relevant dataset

    # Testing case 1: Verify the function returns a string
    print("Testing case [1/2] started.")
    genre = detect_movie_genre(video_path)
    assert isinstance(genre, str), f"Test case [1/2] failed: Expected return type str, got {type(genre)}"

    # Testing case 2: Verify the function raises a FileNotFoundError for a non-existing video file
    print("Testing case [2/2] started.")
    try:
        detect_movie_genre('non_exist_video.mp4')
        assert False, f"Test case [2/2] failed: FileNotFoundError was not raised"
    except FileNotFoundError:
        pass  # Expected exception
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_movie_genre()