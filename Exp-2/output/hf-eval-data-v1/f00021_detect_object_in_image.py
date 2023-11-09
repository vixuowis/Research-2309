import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


def detect_object_in_image(url: str, texts: list):
    '''
    Function to detect objects in an image based on textual description using OwlViTForObjectDetection model.
    Args:
    url : str : URL of the image
    texts : list : List of text queries
    Returns:
    results : dict : Detection results
    '''
    # Load the OwlViTProcessor and OwlViTForObjectDetection model
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')

    # Download the image from the provided URL and open it
    image = Image.open(requests.get(url, stream=True).raw)

    # Pre-process the text query and the image
    inputs = processor(text=texts, images=image, return_tensors='pt')

    # Pass the processed inputs to the model to obtain object detection results
    outputs = model(**inputs)

    # Post-process the results to get object detection information
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results