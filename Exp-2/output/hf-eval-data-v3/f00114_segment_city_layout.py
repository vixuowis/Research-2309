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
    # Load the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')

    # Load the image from the URL
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image = Image.open(response.raw)

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Compute the segmentation
    outputs = model(**inputs)

    return outputs.logits

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