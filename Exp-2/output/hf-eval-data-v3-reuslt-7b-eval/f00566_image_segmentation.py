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
    
    try:
        
        # load the MaskFormer model
        feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
        model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
        
        # download image from the given url
        res = requests.get(image_url)  
        img = Image.open(BytesIO(res.content)) 
         
        # get the segmentation map and draw bounding boxes around recognized objects using MaskFormer model
        segmentator = FeatureExtractorSegmentator(feature_extractor, model)
        
        panoptic_seg = segmentator.segment(img)    
        output_dict = {}
        output_dict['segmap'] = panoptic_seg['panoptic_seg']
        
    except Exception as e:
                
        print('Failed to generate the segmentation map for ', image_url, ' due to an internal error.')  
        return None
    
    finally:
        del feature_extractor, model, img, segmentator
            
    return output_dict


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