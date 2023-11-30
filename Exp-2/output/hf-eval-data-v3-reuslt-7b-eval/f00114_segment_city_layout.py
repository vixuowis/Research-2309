# function_import --------------------

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image
import requests

# function_code --------------------

def segment_city_layout(image_url):
    """
    This function takes an image URL of a city layout and returns the segmented image using a pre-trained model.
    The model used is 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024' from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the city layout image.

    Returns:
        torch.Tensor: The segmented image.

    Raises:
        requests.exceptions.RequestException: If there is a problem with the network connection.
        requests.exceptions.HTTPError: If there is an HTTP error.
        requests.exceptions.Timeout: If the request times out.
        requests.exceptions.TooManyRedirects: If the request exceeds the configured number of maximum redirections.
    """
    
    # load model -------------------------------

    segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    
    # load image -------------------------------

    try:
        response = requests.get(image_url, timeout=5)
    except (requests.exceptions.RequestException):
        raise requests.exceptions.RequestException("There was a problem with the network connection.")
    if response.status_code != 200:
        raise requests.exceptions.HTTPError(f"The HTTP error code is {response.status_code}.")
    
    try:
        image = Image.open(BytesIO(response.content))
    except (OSError):
        raise requests.exceptions.Timeout("The request timed out.")
    
    # preprocess image -------------------------------

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # run model -------------------------------
    
    outputs = segformer_model(**inputs)
    
    # postprocess image -------------------------------
    
    masked_image = inputs['pixel_values'].clone()
    for i in range(masked_image.size(-1)):
        masked_image[:, :, i] *= outputs.logits[0, 4*i:4*(i+1)] > -5
    
    return masked_image


# test_function_code --------------------

def test_segment_city_layout():
    """Tests the `segment_city_layout` function."""
    # Test with a city layout image
    image_url = 'https://placekitten.com/200/300'
    output = segment_city_layout(image_url)
    assert output is not None, 'The output should not be None.'
    assert output.shape[0] == 1, 'The output shape should be (1, num_classes, height, width).'

    # Test with another city layout image
    image_url = 'https://placekitten.com/200/300'
    output = segment_city_layout(image_url)
    assert output is not None, 'The output should not be None.'
    assert output.shape[0] == 1, 'The output shape should be (1, num_classes, height, width).'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_segment_city_layout()