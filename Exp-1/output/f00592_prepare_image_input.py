from typing import *
from torchvision.transforms import functional as F
from PIL import Image
import torch


def prepare_image_input(image, image_processor):
    
    '''
    Prepare the image input for the model using the `image_processor` that will take care of the necessary image transformations
    such as resizing and normalization:

    :param image: The input image
    :param image_processor: The image processor
    :return: The prepared image input as pixel values
    '''
    
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    return pixel_values
