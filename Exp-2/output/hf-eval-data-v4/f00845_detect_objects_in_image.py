# requirements_file --------------------

!pip install -U torch transformers Pillow requests

# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects_in_image(image_url, text_descriptions):
    """
    Detect objects in an image based on provided text descriptions using the OwlViT model.

    Args:
    image_url (str): URL of the image to process.
    text_descriptions (list of str): List of text phrases describing objects to detect.

    Returns:
    list of tuple: Each tuple contains (detected object text, confidence score, bounding box).
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=text_descriptions, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    score_threshold = 0.1

    detected_objects = []
    for i, result in enumerate(results):
        boxes, scores, labels = result['boxes'], result['scores'], result['labels']
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                detected_objects.append((text_descriptions[label], round(score.item(), 3), box))
    return detected_objects

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    text_descriptions = ['a photo of a cat', 'a photo of a dog']

    # Test case: Detect objects using a known image URL and text descriptions
    print("Testing object detection started.")
    results = detect_objects_in_image(image_url, text_descriptions)
    assert len(results) > 0, f"Test failed: Expected to detect at least one object, but got {results}"

    # Additional test cases can be added as needed

    print("Testing finished.")

# Run the test function
test_detect_objects_in_image()