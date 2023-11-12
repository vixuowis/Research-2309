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
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-tiny-coco')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-tiny-coco')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return result

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