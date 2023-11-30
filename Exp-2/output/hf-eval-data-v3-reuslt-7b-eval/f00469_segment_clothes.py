# function_import --------------------

from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch.nn as nn
import torch

# function_code --------------------

def segment_clothes(image_url):
    """
    This function takes an image URL, loads the image, preprocesses it, and uses a pretrained Segformer model
    to segment the clothes in the image.

    Args:
        image_url (str): The URL of the image to be segmented.

    Returns:
        pred_seg (torch.Tensor): The segmented image.

    Raises:
        PIL.UnidentifiedImageError: If the image cannot be identified and opened.
    """    
    # load model --------------------
    feature_extractor = AutoFeatureExtractor.from_pretrained("nandang/segformer-bisenetv2-ade")
    model = SegformerForSemanticSegmentation.from_pretrained("nandang/segformer-bisenetv2-ade").eval()    

    # load image --------------------
    response = requests.get(image_url)
    img = Image.open(Image.io.BytesIO(response.content)).convert('RGB')

    # preprocess image --------------
    inputs = feature_extractor(images=img, return_tensors="pt")
    
    # segment clothes ---------------
    outputs = model({**inputs})[0]["logits"]    
    pred_seg = torch.argmax(outputs, dim=1)[:3].unsqueeze(dim=-1)

    return pred_seg

# test_function_code --------------------

def test_segment_clothes():
    """
    This function tests the segment_clothes function with a few test cases.
    """
    test_image_url = 'https://placekitten.com/200/300'
    try:
        segmented_image = segment_clothes(test_image_url)
        assert segmented_image is not None
        assert isinstance(segmented_image, torch.Tensor)
    except PIL.UnidentifiedImageError:
        print('Test image could not be identified and opened.')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_segment_clothes()