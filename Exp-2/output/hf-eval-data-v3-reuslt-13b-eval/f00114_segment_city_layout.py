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

    # Get image
    try:
      response = requests.get(image_url)
      response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("There was an error getting the image from the URL.")
        raise e
    
    # Convert to PIL Image and resize to required dimensions (1024x1024) using BICUBIC interpolation
    img = Image.open(response.content).resize((1024, 1024), Image.BICUBIC)

    # Load pre-trained model and feature extractor from Hugging Face Transformers
    segmentation_model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    feature_extractor = SegformerFeatureExtractor(size=(1024, 1024), do_resize=False)
    
    # Predict the segmentation map and convert to PIL Image
    outputs = segmentation_model([feature_extractor(images=img, return_tensors="pt")['pixel_values']])
    predicted_masks = torch.argmax(outputs.logits[0], axis=-1).detach().numpy()
    imgs = Image.fromarray(predicted_masks)

    # Convert to binary black and white image (using threshold of 0.25 as the mask contains some values around 0.25)
    bw_image = imgs.point(lambda p: p > 1 and 255, '1')

    return bw_image

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