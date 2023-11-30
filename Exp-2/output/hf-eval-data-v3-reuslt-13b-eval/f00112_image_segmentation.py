# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def image_segmentation(image_url):
    """
    This function segments an image using the SegformerForSemanticSegmentation model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to be segmented.

    Returns:
        logits (torch.Tensor): The output logits from the segmentation model.
    """
    
    url_image = requests.get(image_url)
    img = Image.open(BytesIO(url_image.content))

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to("cuda")
    
    inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(**inputs)
        
    logits = outputs.logits[0]

    return logits

# test_function_code --------------------

def test_image_segmentation():
    """
    This function tests the image_segmentation function with different test cases.
    """
    test_case_1 = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    assert image_segmentation(test_case_1) is not None
    test_case_2 = 'https://placekitten.com/200/300'
    assert image_segmentation(test_case_2) is not None
    test_case_3 = 'https://placekitten.com/500/700'
    assert image_segmentation(test_case_3) is not None
    return 'All Tests Passed'


# call_test_function_code --------------------

test_image_segmentation()