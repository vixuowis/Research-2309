from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import torch


def video_classification(video):
    """
    This function classifies the input video using the pretrained model 'MCG-NJU/videomae-base'.
    
    Parameters:
    video (list): A list of frames of the video to be classified.
    
    Returns:
    loss (torch.Tensor): The loss value of the model.
    """
    # Number of frames in the video
    num_frames = 16
    
    # Create a preprocessor using the VideoMAEImageProcessor with the same model name
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    
    # Load the 'MCG-NJU/videomae-base' model and create the video classification model
    model = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base')
    
    # Preprocess the video frames using the preprocessor and extract pixel values
    pixel_values = processor(video, return_tensors='pt').pixel_values
    
    # Calculate the number of patches per frame
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    
    # Calculate the sequence length
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    
    # Generate a boolean mask for the positions
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
    
    # Pass the preprocessed frames as input to the model to get the video classification results
    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    
    # Get the loss value
    loss = outputs.loss
    
    return loss