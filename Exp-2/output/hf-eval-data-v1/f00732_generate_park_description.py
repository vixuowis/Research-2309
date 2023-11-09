from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

def generate_park_description(img_url):
    '''
    This function generates a description of an image of a park.
    It uses the BlipForConditionalGeneration model from Hugging Face Transformers.
    
    Args:
    img_url (str): The URL or local path of the image.
    
    Returns:
    str: The generated description of the image.
    '''
    # Initialize the image captioning model with pre-trained weights
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    
    # Load the image
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    
    # Preprocess the image
    inputs = processor(raw_image, return_tensors='pt')
    
    # Generate a description of the image
    out = model.generate(**inputs)
    
    # Decode the generated description
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption