# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects(image_url, texts, score_threshold=0.1):
    """
    Detect objects in an image using the OwlViTForObjectDetection model from Hugging Face Transformers.

    Args:
        image_url (str): URL of the image to process.
        texts (list): List of text descriptions to detect in the image.
        score_threshold (float, optional): Confidence score threshold for object detection. Defaults to 0.1.

    Returns:
        None. Prints the detected objects, their confidence scores, and bounding box locations.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    for i, text in enumerate(texts):
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                print(f"Detected {text} with confidence {round(score.item(), 3)} at location {box}")

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a cat', 'a photo of a dog']
    detect_objects(image_url, texts)

# call_test_function_code --------------------

test_detect_objects()