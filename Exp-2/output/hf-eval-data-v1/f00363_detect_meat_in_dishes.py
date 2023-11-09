from PIL import Image
import requests
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


def detect_meat_in_dishes(image_url):
    """
    This function detects if there is any meat in the dishes from the image provided.
    It uses the OwlViTForObjectDetection model from Hugging Face Transformers.
    
    Args:
    image_url (str): The URL of the image containing the dishes.
    
    Returns:
    bool: True if meat is detected, False otherwise.
    """
    # Instantiate the processor and model for object detection
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')
    
    # Open the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Define the text queries
    texts = ['vegan food', 'meat']
    
    # Process the input
    inputs = processor(text=texts, images=image, return_tensors='pt')
    
    # Execute the model to get the outputs
    outputs = model(**inputs)
    
    # Post-process the outputs to obtain the final results
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    
    # Analyze the results to determine if any of the detected objects suggest the presence of meat
    for result in results:
        if 'meat' in result['labels']:
            return True
    return False