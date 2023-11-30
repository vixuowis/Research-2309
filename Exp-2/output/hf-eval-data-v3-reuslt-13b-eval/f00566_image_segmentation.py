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

    # Load pre-trained MaskFormer model for instance segmentation from the hub.
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        "facebook/maskformer-swin-base-ade",
        num_labels=171,  # this is only needed if you want to use your own labels with `from_config` below
    )
    model = model.to("cuda")
    
    # Load pre-trained feature extractor that is used by the model above.
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
        "facebook/maskformer-swin-base-ade"
    )

    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Run the image through the pre-trained MaskFormer model to get predictions for object boundaries and masks.
    outputs = model(**feature_extractor(images=[image], return_tensors="pt"))

    predicted_panoptic = outputs.prediction[0].argmax(dim=0).cpu().detach()
    
    # The dictionary contains the image with recognized objects and their boundaries drawn over it in png format.
    result = {'image' : feature_extractor.to_pil_image(predicted_panoptic, image)} 
    
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