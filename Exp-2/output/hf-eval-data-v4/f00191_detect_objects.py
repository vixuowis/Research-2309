# requirements_file --------------------

!pip install -U transformers torch PIL requests

# function_import --------------------

from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects(image_url: str):
    """
    Detect objects in an image using the deformable-detr model.

    Parameters:
    - image_url: str - The URL of the image to perform detection on.

    Returns:
    - A list of detected object labels.
    """

    # Load and open the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Initialize the processor and model
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')

    # Process the image and convert to torch tensor
    inputs = processor(images=image, return_tensors='pt')

    # Get model output
    outputs = model(**inputs)

    # Extract the labels of detected objects
    labels = outputs.logits.argmax(-1).squeeze().tolist()

    return labels

# test_function_code --------------------

def test_detect_objects():
    print("Testing started.")

    # Test case 1: Check if the function returns a list
    print("Testing case [1/1] started.")
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    detected_objects = detect_objects(sample_image_url)
    assert type(detected_objects) is list, f"Test case [1/1] failed: Function should return a list, got {type(detected_objects)}"
    print("Testing case [1/1] passed.")

    # Additional tests could include:
    # - Mocking the model's output to ensure the rest of function works correctly.
    # - Checking the function with different image URLs.
    # - Ensuring the list contains integers (class labels).
    
    print("Testing finished.")

# Run the test function
test_detect_objects()