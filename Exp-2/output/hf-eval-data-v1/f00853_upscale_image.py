from transformers import Swin2SRForConditionalGeneration
import torch


def upscale_image(low_resolution_tensor):
    """
    Function to upscale a low-resolution image to twice its size using a pretrained Swin2SR model.
    
    Args:
        low_resolution_tensor (torch.Tensor): The low-resolution image tensor.
    
    Returns:
        torch.Tensor: The upscaled image tensor.
    """
    # Load the pretrained Swin2SR model
    model = Swin2SRForConditionalGeneration.from_pretrained('conde/Swin2SR-lightweight-x2-64')
    
    # Pass the tensor through the model to obtain the upscaled image
    upscaled_tensor = model(low_resolution_tensor)
    
    return upscaled_tensor