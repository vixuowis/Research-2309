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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = 'microsoft/videmae-large' # pretrained model on Clevrer dataset
    
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_id)
    feature_extractor = feature_extractor.to(device)

    max_frames = 288
    frame_rate = 30

    video_reader = VideoReader(file_path, ctx=cpu(0)) # create a decord video reader object
    
    clip = []
    for frame in range(int(min((video_reader.total_frames // frame_sample_rate), max_frames) * frame_sample_rate)):
        clip.append(video_reader[frame::frame_sample_rate].asnumpy()) # extract video frames and store them as numpy array
    
    if clip_len < len(clip): 
        start = np.random.randint(0, len(clip) - clip_len)
        end = start + clip_len
        
    else:
        start = 0
        end = clip_len - (len(clip) % clip_len)
    
    video_array = np.stack(clip[start:end]) # convert the numpy array frames to a tensor
    video_tensor = torch.from_numpy(video_array).to(device) / 255 # normalize the values of pixels between [0,1]
    
    inputs = feature_extractor(video_tensor, sampling_strategy='random', frames_per_second=frame_rate, max_frames=max_frames)['input_values'][None] # extract features from video clip
    
    model = VideoMAEForVideoClassification.from_pretrained(model_id).to(device)
    
    with torch.no_grad():
        outputs = model(inputs)[0].cpu().numpy()
        
    action_dict = {'action': 'None', 'probability': 1}
    for i in range(len(outputs)):

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