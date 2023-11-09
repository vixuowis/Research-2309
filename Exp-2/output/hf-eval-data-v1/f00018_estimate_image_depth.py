from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests


def estimate_image_depth(image_url):
    """
    This function estimates the depth of an image using a pretrained model from Hugging Face Transformers.
    The model used is 'Intel/dpt-large', which is specifically designed for monocular depth estimation.
    
    Parameters:
    image_url (str): The URL of the image to be analyzed.
    
    Returns:
    depth (Image): An image representing the estimated depth of the input image.
    """
    # Load the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Load the image processor and the depth estimation model
    processor = DPTImageProcessor.from_pretrained('Intel/dpt-large')
    model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
    
    # Preprocess the image and pass it to the model
    inputs = processor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Format the output into an interpretable format
    prediction = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=image.size[::-1], mode='bicubic', align_corners=False)
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')
    depth = Image.fromarray(formatted)
    
    return depth