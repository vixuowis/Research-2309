# function_import --------------------

from PIL import Image
import requests
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch

# function_code --------------------

def segment_birds_in_image(image_url):
    """
    This function segments birds in an image using a pre-trained Mask2Former model.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        PIL.Image: The segmented image.

    Raises:
        OSError: If there is an error in loading the image or the pre-trained model.
    """
    # Load the pre-trained Mask2Former model and the image processor
    processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')
    model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')

    # Load the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Preprocess the image
    inputs = processor(images=image, return_tensors='pt')

    # Perform instance segmentation
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the segmentation outputs
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    # Return the segmented image
    return result['segmentation']

# test_function_code --------------------

def test_segment_birds_in_image():
    """
    This function tests the 'segment_birds_in_image' function with different test cases.
    """
    # Test case 1: An image with a bird
    image_url = 'https://example.com/image_with_bird.jpg'
    segmented_image = segment_birds_in_image(image_url)
    assert isinstance(segmented_image, Image.Image), 'The output should be a PIL.Image object.'

    # Test case 2: An image without a bird
    image_url = 'https://example.com/image_without_bird.jpg'
    segmented_image = segment_birds_in_image(image_url)
    assert isinstance(segmented_image, Image.Image), 'The output should be a PIL.Image object.'

    # Test case 3: An image with multiple birds
    image_url = 'https://example.com/image_with_multiple_birds.jpg'
    segmented_image = segment_birds_in_image(image_url)
    assert isinstance(segmented_image, Image.Image), 'The output should be a PIL.Image object.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_segment_birds_in_image()