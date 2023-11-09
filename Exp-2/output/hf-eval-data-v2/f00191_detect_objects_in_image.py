# function_import --------------------

from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image(image_url):
    """
    Detect objects in an image using the DeformableDetrForObjectDetection model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to process.

    Returns:
        dict: The outputs of the DeformableDetrForObjectDetection model.
    """
    # Load the image from the given URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Instantiate the AutoImageProcessor and the DeformableDetrForObjectDetection model
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')

    # Process the image and detect objects
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    return outputs

# test_function_code --------------------

def test_detect_objects_in_image():
    """
    Test the detect_objects_in_image function with a sample image from the COCO 2017 dataset.
    """
    # Define a sample image URL from the COCO 2017 dataset
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Call the function with the sample image URL
    outputs = detect_objects_in_image(sample_image_url)

    # Assert that the outputs are not None
    assert outputs is not None, 'The function did not return any outputs.'

    # Assert that the outputs are of the expected type
    assert isinstance(outputs, dict), 'The function did not return a dictionary.'

# call_test_function_code --------------------

test_detect_objects_in_image()