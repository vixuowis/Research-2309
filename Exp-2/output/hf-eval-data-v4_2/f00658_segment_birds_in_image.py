# requirements_file --------------------

!pip install -U torch transformers Pillow requests

# function_import --------------------

from PIL import Image
import requests
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch

# function_code --------------------

def segment_birds_in_image(image_url):
    """Segments birds in an image using a pre-trained Mask2Former model.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        dict: A dictionary containing the segmented image map.

    Raises:
        Exception: Any exception thrown by underlying library calls.
    """

    try:
        processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')
        model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(images=image, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_instance_map = result['segmentation']
        return {'segmentation': predicted_instance_map}
    except Exception as e:
        raise e

# test_function_code --------------------

def test_segment_birds_in_image():
    print("Testing started.")
    # Replace with actual image URL containing birds
    test_image_url = 'https://example.com/test_image_with_birds.jpg'

    # Testing case 1: Check if the function returns a dictionary
    print("Testing case [1/1] started.")
    result = segment_birds_in_image(test_image_url)
    assert isinstance(result, dict), f"Test case [1/1] failed: Expected result type dict, got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_birds_in_image()