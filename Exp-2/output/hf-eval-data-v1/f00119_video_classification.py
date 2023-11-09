from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import torch


def video_classification(video):
    '''
    This function classifies the action in a video using the VideoMAE pretrained model.
    
    Parameters:
    video (list): A list of numpy arrays, one for each frame of the video.
    
    Returns:
    loss (torch.Tensor): The loss value of the model output.
    '''
    # Load the VideoMAE pretrained model
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    
    # Create a VideoMAEImageProcessor instance
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short-ssv2')
    
    # Process the video frames
    pixel_values = processor(video, return_tensors='pt').pixel_values
    
    # Configure and prepare the model
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    num_frames = len(video)
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    
    # Pass the processed video to the model and evaluate the output
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    
    return outputs.loss