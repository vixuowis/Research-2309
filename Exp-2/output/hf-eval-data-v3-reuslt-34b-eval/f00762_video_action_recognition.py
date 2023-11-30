# function_import --------------------

from decord import VideoReader, cpu
import torch
import numpy as np
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download

# function_code --------------------

def video_action_recognition(file_path: str, clip_len: int = 16, frame_sample_rate: int = 4):
    '''
    Function to recognize the main action in a video clip using VideoMAE model.
    
    Args:
        file_path (str): Path to the video file.
        clip_len (int, optional): Length of the clip for which the action is to be recognized. Default is 16.
        frame_sample_rate (int, optional): Frame sample rate. Default is 4.
    
    Returns:
        str: The recognized main action in the video clip.
    '''

    # Load feature extractor and model.
    print('Loading...')
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('facebook/video-mae-base-kinetics-600h')
    model = VideoMAEForVideoClassification.from_pretrained('facebook/video-mae-base-kinetics-600h', num_labels=400)

    # Load and preprocess frames from video file.
    with open(file_path, 'rb') as f:
        full = np.frombuffer(f.read(), dtype=np.uint8)
    frames = VideoReader(full, ctx=cpu(0))  # change context to gpu if available.
    
    # Sample frames based on frame sample rate and preprocess frames using feature extractor.
    step_size = clip_len * (frame_sample_rate - 1) + 1
    frames_select = frames[::step_size][:clip_len]
    video = []
    
    # Decord is used to decode videos into numerical array representation for preprocessing.
    for i in range(0, clip_len):
        img = frames_select[i].asnumpy()
        img = img[:, :, ::-1]  # RGB to BGR format
        video.append(img)
    
    # Preprocess selected frames using feature extractor and convert list to numpy array.
    inputs = feature_extractor(images=video, return_tensors='pt')['pixel_values']
    print('Loaded...')

    # Prediction on video.
    with torch.no_grad():
        prediction = model(inputs)
    
    # Get label from prediction.
    predicted_class_idx = prediction.logits.argmax(-1).item()
    le = hf_hub_download('facebook/video-mae-base-kinetics-600h', filename='label_map.txt')
    
    with open(le) as f:
        lbl_list = [ll.split()[1][:-1] for ll in f.

# test_function_code --------------------

def test_video_action_recognition():
    '''
    Function to test the video_action_recognition function.
    '''
    file_path = hf_hub_download('archery.mp4')
    assert isinstance(video_action_recognition(file_path), str), 'The function should return a string.'
    assert video_action_recognition(file_path) != '', 'The function should not return an empty string.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_video_action_recognition()