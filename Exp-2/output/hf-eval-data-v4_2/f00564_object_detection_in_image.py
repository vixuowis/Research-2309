# requirements_file --------------------

!pip install -U torch transformers Pillow requests

# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def object_detection_in_image(image_url: str, score_threshold: float = 0.1):
    """
    Detects objects in an image and prints their locations and confidence scores.

    Args:
        image_url: The URL of the image to detect objects in.
        score_threshold: The threshold score for accepting detected objects.

    Returns:
        A dictionary containing the detected objects and their details.

    Raises:
        ValueError: If the image cannot be loaded from the provided URL.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('The image cannot be loaded.') from e

    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')

    texts = ['a photo of something']
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    detections = []
    for i, text in enumerate(texts):
        boxes, scores, labels = results[i]['boxes'], results[i]['scores'], results[i]['labels']
        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                detections.append(
                    {'text': text,
                     'box': [round(b, 2) for b in box.tolist()],
                     'score': round(score.item(), 3)}
                )

    return detections

# test_function_code --------------------

def test_object_detection_in_image():
    print('Testing started.')
    # Load a sample image URL for testing
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Test case 1: Valid image URL
    print('Testing case [1/1] started.')
    try:
        detections = object_detection_in_image(image_url, score_threshold=0.1)
        assert detections, f'Test case [1/1] failed: No detections were found.'
    except ValueError as e:
        assert False, f'Test case [1/1] failed: {e}'
    print('Testing finished.')

# call_test_function_line --------------------

test_object_detection_in_image()