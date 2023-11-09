from PIL import Image
import requests
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def detect_suspicious_objects(url: str, texts: list):
    """
    Detect suspicious objects and people in an image using a zero-shot text-conditioned object detection system.

    Args:
        url (str): The URL of the image to analyze.
        texts (list): A list of text descriptions that represent suspicious objects and people.

    Returns:
        A dictionary containing the results of the object detection.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    return results