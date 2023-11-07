from typing import *
from transformers import pipeline

def depth_estimation(image_url: str) -> List[float]:
    """
    Estimates the depth of each pixel in an image.

    Args:
        image_url (str): The URL of the image.

    Returns:
        List[float]: The depth values of each pixel in the image.
    """
    depth_estimator = pipeline(task="depth-estimation")
    preds = depth_estimator(image_url)
