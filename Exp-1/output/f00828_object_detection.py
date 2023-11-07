from typing import *
from transformers import pipeline


def object_detection(image_url: str) -> List[Dict[str, Union[float, str, Dict[str, int]]]]:
    '''
    Performs object detection on the given image URL.
    
    Args:
        image_url (str): The URL of the image to perform object detection on.
    
    Returns:
        List[Dict[str, Union[float, str, Dict[str, int]]]]: A list of dictionaries, where each dictionary represents an object detected in the image. Each dictionary contains the following keys:
            - 'score' (float): The confidence score of the object detection.
            - 'label' (str): The label of the detected object.
            - 'box' (Dict[str, int]): The bounding box coordinates of the detected object, represented as a dictionary with the keys 'xmin', 'ymin', 'xmax', and 'ymax'.
    '''
    detector = pipeline(task='object-detection')
    preds = detector(image_url)
    preds = [{'score': round(pred['score'], 4), 'label': pred['label'], 'box': pred['box']} for pred in preds]
    return preds
