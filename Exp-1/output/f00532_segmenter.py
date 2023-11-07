from typing import *
from transformers import pipeline

def segmenter(model, image):
    '''
    Instantiate a pipeline for image segmentation with the given model and pass the image to it.
    
    Args:
    - model (str): The name or path of the image segmentation model.
    - image (PIL.Image.Image): The input image.
    
    Returns:
    - List[Dict]: A list of dictionaries containing the segmentation results, each with the keys 'score', 'label', and 'mask'.
    '''
    segmenter = pipeline('image-segmentation', model=model)
    return segmenter(image)
