from typing import *
from PIL import Image

def get_image_label(data):
    '''
    Get the image and label from the data

    Args:
        data (dict): The data containing the image and label

    Returns:
        tuple: A tuple containing the image and label
    '''
    image = data['image']
    label = data['label']
    return image, label
