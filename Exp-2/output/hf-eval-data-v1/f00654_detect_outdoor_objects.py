from transformers import OwlViTForObjectDetection, OwlViTProcessor
from PIL import Image
import requests
import torch


def detect_outdoor_objects(image_url):
    """
    This function detects outdoor objects in an image using the OwlViTForObjectDetection model.
    
    Parameters:
    image_url (str): The URL of the image to process.
    
    Returns:
    dict: The detected objects and their locations.
    """
    # Create an instance of the OwlViTProcessor using the 'google/owlvit-base-patch16' pretrained model.
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    # Create an instance of the OwlViTForObjectDetection model using the pretrained model.
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')
    
    # Open the image from the provided URL.
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Create a list of text queries representing outdoor objects.
    texts = [['a tent', 'a backpack', 'hiking boots', 'a campfire', 'a kayak']]
    
    # Feed the image and the text queries to the processor to preprocess the inputs.
    inputs = processor(text=texts, images=image, return_tensors='pt')
    
    # Pass the preprocessed inputs to the model for object detection.
    outputs = model(**inputs)
    
    # Post-process the outputs to get the final results.
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    
    return results