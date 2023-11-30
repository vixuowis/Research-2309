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
    
    # load the pretrained model
    model = VideoMAEForVideoClassification.from_pretrained("hf-internal-testing/tiny-vit-14x14-mae")
    feature_extractor = VideoMAEFeatureExtractor()
    
    # read the video file
    reader = VideoReader(file_path, ctx=cpu(0))
    
    # get the number of frames in the clip
    num_frames = len(reader)
    
    # calculate the frame index step size for given parameters
    sampling_step_size = int((num_frames - 16 / frame_sample_rate) // (clip_len-1))
    
    # extract the features using VideoMAEFeatureExtractor
    features = feature_extractor(reader, sampling_strategy='random', num_sampling_steps=int(num_frames/sampling_step_size+1), 
                                 device="cpu", stack_dir="row")
    
    # convert the extracted features to tensor
    inputs = torch.tensor(features)
    
    # predict and return the predicted class
    preds = model(inputs).logits
    
    return model.config.id2label[np.argmax(preds)]

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