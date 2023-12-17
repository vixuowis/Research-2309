# requirements_file --------------------

!pip install -U requests Pillow torch transformers

# function_import --------------------

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import requests
import torch

# function_code --------------------

def detect_property_objects(image_url, query_texts):
    """
    Detect objects in a property image with text queries using the OwlViT model.

    Parameters:
        image_url (str): URL of the image to be processed.
        query_texts (list): List containing text queries corresponding to objects.

    Returns:
        list: Detected objects and their associated confidence levels.
    """
    # Load the OwlVit-based model and processor
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')
    
    # Load the image from provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Process the inputs
    inputs = processor(text=[query_texts], images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    
    # Post-process the outputs
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    
    # Extract and return detected objects and their scores
    detected_objects = []
    for result in results[0]:
        detected_objects.append({
            'label': result['score'].item(),
            'score': result['score'].item()
        })
    return detected_objects


# test_function_code --------------------

def test_detect_property_objects():
    print("Testing started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    query_texts = ['a photo of a living room', 'a photo of a kitchen', 'a photo of a bedroom', 'a photo of a bathroom']

    # Test case 1: Detect objects in a property image
    print("Testing case [1/1] started.")
    results = detect_property_objects(image_url, query_texts)
    assert type(results) == list and len(results) > 0, f"Test case [1/1] failed: The function did not return any results."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_detect_property_objects()
