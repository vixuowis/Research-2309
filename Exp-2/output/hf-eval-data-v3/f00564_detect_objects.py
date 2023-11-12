# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects(image_url: str, texts: list) -> None:
    """
    Detect objects in an image and describe their locations.

    Args:
        image_url (str): The URL of the image to process.
        texts (list): List of text descriptions to detect in the image.

    Returns:
        None. Prints the detected objects, their confidence scores, and bounding box locations.

    Raises:
        OSError: If there is an error in loading the model or the image.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    score_threshold = 0.1

    for i, text in enumerate(texts):
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                print(f'Detected {text} with confidence {round(score.item(), 3)} at location {box}')

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    image_url = 'https://placekitten.com/200/300'
    texts = ['a photo of a cat', 'a photo of a dog']
    try:
        detect_objects(image_url, texts)
        print('Test passed')
    except Exception as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_detect_objects()