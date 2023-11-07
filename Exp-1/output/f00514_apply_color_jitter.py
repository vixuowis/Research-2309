from typing import *
from torchvision.transforms import ColorJitter

def apply_color_jitter(image, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1):
    """
    Apply color jitter to an image.
    
    Args:
        image (PIL.Image or numpy.ndarray): The input image.
        brightness (float): How much to jitter brightness.
        contrast (float): How much to jitter contrast.
        saturation (float): How much to jitter saturation.
        hue (float): How much to jitter hue.
    
    Returns:
        PIL.Image or numpy.ndarray: The color jittered image.
    """
    jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    return jitter(image)
