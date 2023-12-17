# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers", "Pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects_in_image(image_url: str, text_queries: list) -> dict:
    """
    Detects objects in an image according to specified text queries using the OwlViT model.

    Args:
        image_url (str): The URL of the image to be analyzed.
        text_queries (list): A list of text phrases describing the objects to be detected.

    Returns:
        dict: A dictionary with detected objects, confidence scores, and bounding boxes.

    Raises:
        ValueError: If the image_url or the text_queries are not valid.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=text_queries, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    score_threshold = 0.1
    detections = []
    for result in results:
        boxes, scores, labels = result['boxes'], result['scores'], result['labels']
        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                detection = {
                    'object': text_queries[label],
                    'confidence': round(score.item(), 3),
                    'box': [round(i, 2) for i in box.tolist()]
                }
                detections.append(detection)
    return {'detections': detections}

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    text_queries = ['a photo of a cat', 'a photo of a dog']

    print("Testing case [1/1] started.")
    result = detect_objects_in_image(image_url, text_queries)
    assert result and 'detections' in result, "Test case failed: No detections found."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_objects_in_image()