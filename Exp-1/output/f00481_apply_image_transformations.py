from typing import *
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def apply_image_transformations(image_processor):
    """
    Apply image transformations to make the model more robust against overfitting.

    Params:
    - image_processor (object): An object containing image processing parameters.

    Returns:
    - transforms (object): An object containing the applied image transformations.
    """
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size['shortest_edge']
        if 'shortest_edge' in image_processor.size
        else (image_processor.size['height'], image_processor.size['width'])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
