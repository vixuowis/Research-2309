# function_import --------------------

from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
from decord import VideoReader
import torch
import numpy as np

# function_code --------------------

def detect_movie_genre(video_filename):
    '''
    Detect the genre of a movie based on its actions.
    
    Args:
        video_filename (str): The path to the video file.
    
    Returns:
        str: The predicted genre of the movie.
    
    Raises:
        FileNotFoundError: If the video file does not exist.
    '''
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
    '''
    Test the detect_movie_genre function.
    '''
    # Test with a video file
    assert detect_movie_genre('path/to/video_file.mp4') == 'Expected Genre'
    # Test with a non-existent video file
    try:
        detect_movie_genre('path/to/non_existent_file.mp4')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Expected a FileNotFoundError.')
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_detect_movie_genre())