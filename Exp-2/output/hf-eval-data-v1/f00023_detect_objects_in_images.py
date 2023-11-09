from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import requests
import torch


def detect_objects_in_images(url, texts):
    '''
    This function detects objects in images using the OwlViTForObjectDetection model from Hugging Face Transformers.
    Args:
    url : str : The URL of the image.
    texts : list : The texts representing the objects of interest.
    Returns:
    results : dict : The detected objects in the image.
    '''
    # Instantiate the OwlViTProcessor and OwlViTForObjectDetection classes
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')

    # Load the image from the URL
    image = Image.open(requests.get(url, stream=True).raw)

    # Process the input data and generate appropriate tensors
    inputs = processor(text=texts, images=image, return_tensors='pt')

    # Pass the input tensors into the OwlViTForObjectDetection model
    outputs = model(**inputs)

    # Decode the detected objects and gather the results
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results