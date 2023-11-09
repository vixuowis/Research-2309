# function_import --------------------

from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

# function_code --------------------

def segment_clothes(image_path):
    """
    This function segments clothes in an image using a pre-trained SegFormer model.

    Args:
        image_path (str): The path to the image file or a URL.

    Returns:
        A matplotlib figure showing the segmented image.
    """
    extractor = AutoFeatureExtractor.from_pretrained('mattmdjaga/segformer_b2_clothes')
    image = Image.open(image_path)
    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    inputs = extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    plt.imshow(pred_seg)
    return plt

# test_function_code --------------------

def test_segment_clothes():
    """
    This function tests the segment_clothes function by using a sample image.
    """
    url = 'https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&amp;ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&amp;w=1000&amp;q=80'
    result = segment_clothes(url)
    assert isinstance(result, type(plt)), 'The result should be a matplotlib figure.'

# call_test_function_code --------------------

test_segment_clothes()