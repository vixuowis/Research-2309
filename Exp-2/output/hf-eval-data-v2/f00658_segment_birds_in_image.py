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
    This function tests the 'segment_birds_in_image' function by using a sample image URL.
    """
    # Define a sample image URL
    image_url = 'https://example.com/image_with_birds.jpg'

    # Call the 'segment_birds_in_image' function
    segmented_image = segment_birds_in_image(image_url)

    # Assert that the function returns an instance of PIL.Image
    assert isinstance(segmented_image, Image.Image)

# call_test_function_code --------------------

test_segment_birds_in_image()