from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
from decord import VideoReader
import torch
import numpy as np

def detect_movie_genre(video_filename):
    '''
    This function detects the genre of a movie based on its actions.
    It uses a pre-trained model from Hugging Face Transformers for video action recognition.
    
    Parameters:
    video_filename (str): The path to the video file.
    
    Returns:
    str: The predicted genre of the movie.
    '''
    # Load the video data
    videoreader = VideoReader(video_filename)

    # Load the pre-trained model and feature extractor
    model = VideoMAEForVideoClassification.from_pretrained('nateraw/videomae-base-finetuned-ucf101')
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('nateraw/videomae-base-finetuned-ucf101')

    # Sample every 4 frames from the video
    frames = videoreader.get_batch(list(range(0, len(videoreader), 4)))

    # Extract features from the frames
    inputs = feature_extractor(list(frames.asnumpy()), return_tensors='pt')

    # Get the model's output
    outputs = model(**inputs)

    # Get the predicted label
    predicted_label = outputs.logits.argmax(-1).item()

    # Return the predicted genre
    return model.config.id2label[predicted_label]