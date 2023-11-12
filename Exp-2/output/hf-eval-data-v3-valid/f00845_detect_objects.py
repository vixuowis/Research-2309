# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects(url: str, texts: list, model_name: str = 'google/owlvit-large-patch14', score_threshold: float = 0.1):
    """
    Detect objects in an image based on specific text phrases using the OwlViT model.

    Args:
        url (str): The URL of the image.
        texts (list): A list of text descriptions.
        model_name (str, optional): The name of the OwlViT model. Defaults to 'google/owlvit-large-patch14'.
        score_threshold (float, optional): The score threshold for filtering detections. Defaults to 0.1.

    Returns:
        None. Prints the detected objects, their confidence scores, and bounding box locations.
    """
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    for i, result in enumerate(results):
        boxes, scores, labels = result['boxes'], result['scores'], result['labels']
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                print(f'Detected {texts[label]} with confidence {round(score.item(), 3)} at location {box}')

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a cat', 'a photo of a dog']
    try:
        detect_objects(url, texts)
        print('Test passed.')
    except Exception as e:
        print('Test failed. Error: ', e)

# call_test_function_code --------------------

test_detect_objects()