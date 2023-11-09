import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel


def classify_image(url, candidate_labels):
    '''
    This function classifies an image into one of the given categories using the ALIGN model.
    
    Parameters:
    url (str): The URL of the image to be classified.
    candidate_labels (list): The list of possible categories for the image.
    
    Returns:
    list: The probabilities of the image belonging to each category.
    '''
    # Load the AlignProcessor and AlignModel
    processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
    model = AlignModel.from_pretrained('kakaobrain/align-base')
    
    # Load the image from the given URL
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Create inputs for the model
    inputs = processor(text=candidate_labels, images=image, return_tensors='pt')
    
    # Classify the image
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate the probabilities of the image belonging to each category
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    return probs.tolist()