from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
import numpy as np
import torch


def classify_sports_videos(video):
    '''
    This function classifies actions of athletes in sports videos using the pre-trained 'MCG-NJU/videomae-base-short-ssv2' model.
    
    Parameters:
    video (list): A list of frames representing the video to be classified.
    
    Returns:
    outputs (BaseModelOutputWithPoolingAndCrossAttentions): The output from the model which includes the loss and the last hidden state.
    '''
    
    # Load the pre-trained model and feature extractor
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    
    # Process the video file into frames and feed them into the feature_extractor
    pixel_values = feature_extractor(video, return_tensors='pt').pixel_values
    
    # Calculate the number of patches per frame and the sequence length
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    num_frames = len(video)
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    
    # Generate a boolean mask for the positions to be masked
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    
    # Use the model to analyze the input frames and classify the actions/activities
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    
    return outputs