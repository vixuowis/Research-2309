# requirements_file --------------------

!pip install -U requests Pillow torch transformers

# function_import --------------------

from transformers import OwlViTForObjectDetection, OwlViTProcessor
from PIL import Image
import requests
import torch

# function_code --------------------

def identify_outdoor_objects(image_url, text_queries):
    """
    Identify objects in an image that are related to outdoor activities using the OwlViT model.

    :param image_url: The URL of the image to be processed.
    :param text_queries: A list of text queries representing objects related to outdoor activities.
    :return: A dictionary containing the detected objects and their scores.
    """
    # Load the processor and the model
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')

    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Preprocess inputs
    inputs = processor(text=[text_queries], images=image, return_tensors='pt')
    outputs = model(**inputs)

    # Post-process outputs
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    # Extract detected objects and scores
    detected_objects = [{'label': res['text'], 'score': res['score']} for res in results['answers']]

    return detected_objects

# test_function_code --------------------

def test_identify_outdoor_objects():
    print("Testing started.")
    # Use a sample image URL
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    text_queries = ['a tent', 'a backpack', 'hiking boots', 'a campfire', 'a kayak']

    # Test case 1: Check if function returns a list
    print("Testing case [1/3] started.")
    results = identify_outdoor_objects(image_url, text_queries)
    assert isinstance(results, list), f"Test case [1/3] failed: Function should return a list."

    # Test case 2: Check if returned list contains dictionaries
    print("Testing case [2/3] started.")
    valid_types = all(isinstance(item, dict) for item in results)
    assert valid_types, f"Test case [2/3] failed: Each item in the result should be a dictionary."

    # Test case 3: Check if dictionaries have 'label' and 'score' keys
    print("Testing case [3/3] started.")
    valid_keys = all('label' in item and 'score' in item for item in results)
    assert valid_keys, f"Test case [3/3] failed: Each dictionary should have 'label' and 'score' keys."
    print("Testing finished.")