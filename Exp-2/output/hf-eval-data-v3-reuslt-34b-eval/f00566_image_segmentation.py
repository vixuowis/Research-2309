# function_import --------------------

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# function_code --------------------

def image_segmentation(image_url):
    '''
    Recognize the objects in a given image and draw a boundary around them using MaskFormer model.
    
    Args:
        image_url (str): The url of the image to be processed.
    
    Returns:
        dict: A dictionary containing the predicted panoptic map with recognized objects and their boundaries.
    '''

    # load maskformer feature extractor and model pretrained on coco dataset
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
    
    # download image from url
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # call function of feature extractor to normalize the input and convert it into a pytorch tensor
    encoding = feature_extractor(image, return_tensors="pt")
    
    # create model and load pretrained weights
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")
    
    # forward pass for prediction
    outputs = model(**encoding, output_attentions=False)
        
    # return panoptic map with recognized objects and their boundaries (as tensor)
    return outputs.pixel_values


# test_function_code --------------------

def test_image_segmentation():
    '''
    Test the image_segmentation function with different test cases.
    '''
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = image_segmentation(test_image_url)
    assert 'segmentation' in result, 'Test Case 1 Failed'
    
    test_image_url = 'https://placekitten.com/200/300'
    result = image_segmentation(test_image_url)
    assert 'segmentation' in result, 'Test Case 2 Failed'
    
    print('All Tests Passed')


# call_test_function_code --------------------

test_image_segmentation()