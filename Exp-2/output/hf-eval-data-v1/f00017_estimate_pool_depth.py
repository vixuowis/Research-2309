import torch
from transformers import AutoModel


def estimate_pool_depth(underwater_photo):
    """
    This function estimates the depth of a pool given an underwater photo.
    It uses a pre-trained model from Hugging Face Transformers library.
    
    Parameters:
    underwater_photo (str): The path to the underwater photo.
    
    Returns:
    float: The estimated depth of the pool.
    """
    # Load the pre-trained model
    model = AutoModel.from_pretrained('hf-tiny-model-private/tiny-random-GLPNForDepthEstimation')
    
    # Pre-process underwater photo and convert to tensor
    underwater_photo_tensor = preprocess_underwater_photo(underwater_photo)
    
    # Get depth estimation from the model
    depth_estimation = model(underwater_photo_tensor)
    
    return depth_estimation