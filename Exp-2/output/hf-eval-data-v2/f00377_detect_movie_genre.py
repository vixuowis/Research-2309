# function_import --------------------

from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
from decord import VideoReader
import torch
import numpy as np

# function_code --------------------

def detect_movie_genre(video_filename):
    """
    Detect the genre of a movie based on its actions.

    Args:
        video_filename (str): The path to the video file.

    Returns:
        str: The predicted genre of the movie.
    """
    videoreader = VideoReader(video_filename)

    model = VideoMAEForVideoClassification.from_pretrained('nateraw/videomae-base-finetuned-ucf101')
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('nateraw/videomae-base-finetuned-ucf101')

    frames = videoreader.get_batch(list(range(0, len(videoreader), 4))) # Sample every 4 frames
    inputs = feature_extractor(list(frames.asnumpy()), return_tensors='pt')
    outputs = model(**inputs)

    predicted_label = outputs.logits.argmax(-1).item()

    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_detect_movie_genre():
    """
    Test the detect_movie_genre function.
    """
    video_filename = 'path/to/test_video_file.mp4' # replace with the path to your test video file
    predicted_genre = detect_movie_genre(video_filename)
    print(f'The predicted genre of the movie is: {predicted_genre}')
    assert isinstance(predicted_genre, str), 'The predicted genre should be a string.'

# call_test_function_code --------------------

test_detect_movie_genre()