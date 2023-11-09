from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel


def classify_image(image_url, texts):
    """
    This function classifies an image based on semantic similarity to the provided texts.
    It uses a pre-trained Chinese CLIP model from Hugging Face Transformers.
    
    Parameters:
    image_url (str): The URL of the image to classify.
    texts (list): A list of text descriptions to compare the image against.
    
    Returns:
    list: A list of probabilities corresponding to the semantic similarity of the image to each text description.
    """
    # Load the pre-trained model and processor
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    
    # Open the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Process the image and text descriptions
    inputs = processor(text=texts, images=image, return_tensors='pt', padding=True)
    
    # Classify the image
    outputs = model(**inputs)
    
    # Calculate the probabilities
    probs = outputs.logits_per_image.softmax(dim=1)
    
    return probs.tolist()