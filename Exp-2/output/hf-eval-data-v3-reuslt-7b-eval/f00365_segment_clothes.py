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
    
    # Load our pre-trained feature extractor and semantic segmentation model
    feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to('cuda')
    
    # Load the image from the given path
    img = Image.open(requests.get(image_path, stream=True).raw if str(image_path).startswith('http') else image_path)

    # Apply our feature extractor to get the features of the input image
    inputs = feature_extractor(images=img, return_tensors="pt")
    
    # Feed those features into the model and obtain the logits (i.e., the predictions before softmax)
    outputs = model(**inputs)['logits']
    
    # Get the highest scoring segmentation from the model
    predicted_segmentation = nn.functional.softmax(outputs, dim=1).argmax(dim=1)

    # Create a PIL image object from the segmentation and plot it using matplotlib
    fig, ax = plt.subplots()
    seg = Image.open(requests.get('https://github.com/huggingface/transformers/raw/master/tests/dummy-data/samples/image-segmentation/ade20k_val_00000003.jpg', stream=True).raw if str(image_path).startswith('http') else 'ade20k_val_00000003.jpg')
    ax.imshow((predicted_segmentation == 16).numpy().astype(float))
    
    return fig

# test_function_code --------------------

def test_segment_clothes():
    """
    This function tests the segment_clothes function with a few test cases.
    """
    # Test case 1: An image of a person wearing clothes
    url1 = 'https://placekitten.com/200/300'
    result1 = segment_clothes(url1)
    assert isinstance(result1, type(plt)), 'Test Case 1 Failed'

    # Test case 2: Another image of a person wearing clothes
    url2 = 'https://placekitten.com/400/600'
    result2 = segment_clothes(url2)
    assert isinstance(result2, type(plt)), 'Test Case 2 Failed'

    # Test case 3: Yet another image of a person wearing clothes
    url3 = 'https://placekitten.com/800/1200'
    result3 = segment_clothes(url3)
    assert isinstance(result3, type(plt)), 'Test Case 3 Failed'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_segment_clothes()