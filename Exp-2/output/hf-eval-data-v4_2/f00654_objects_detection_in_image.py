# requirements_file --------------------

!pip install -U requests Pillow torch transformers

# function_import --------------------

from transformers import OwlViTForObjectDetection, OwlViTProcessor
from PIL import Image
import requests

# function_code --------------------

def objects_detection_in_image(image_url, queries):
    """Detect objects related to outdoor activities in images using the OwlViT model.

    Args:
        image_url (str): The URL of the image to analyze.
        queries (list): A list of text queries representing outdoor objects.

    Returns:
        dict: The detected objects and their details.

    Raises:
        ValueError: If image can't be loaded from URL or other processing errors.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except IOError as e:
        raise ValueError('Failed to load image from the provided URL.') from e

    inputs = processor(text=[queries], images=image, return_tensors='pt')
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results

# test_function_code --------------------

def test_objects_detection_in_image():
    print("Testing started.")
    # Test case 1: Valid outdoor image with common objects
    print("Testing case [1/1] started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    queries = ['a tent', 'a backpack', 'hiking boots', 'a campfire', 'a kayak']
    try:
        results = objects_detection_in_image(image_url, queries)
        assert "boxes" in results, "Results should contain detected boxes."
        assert len(results["boxes"]) > 0, "There should be at least one detected box."
    except ValueError as e:
        assert False, f"Test case [1/1] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_objects_detection_in_image()