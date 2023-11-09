from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

def classify_image(img_url):
    """
    Classify an image using a fine-tuned CLIP model.

    Args:
        img_url (str): The URL of the image to classify.

    Returns:
        dict: A dictionary where the keys are the labels and the values are the corresponding probabilities.
    """
    # Load the fine-tuned CLIP model and the processor
    model = CLIPModel.from_pretrained('flax-community/clip-rsicd-v2')
    processor = CLIPProcessor.from_pretrained('flax-community/clip-rsicd-v2')

    # Open the image
    image = Image.open(requests.get(img_url, stream=True).raw)

    # Define the labels
    labels = ['residential area', 'playground', 'stadium', 'forest', 'airport']

    # Process the text and the image to create input tensors
    inputs = processor(text=[f'a photo of a {l}' for l in labels], images=image, return_tensors='pt', padding=True)

    # Pass the input tensors to the model
    outputs = model(**inputs)

    # Get the logits per image
    logits_per_image = outputs.logits_per_image

    # Apply the softmax function to get the probabilities
    probs = logits_per_image.softmax(dim=1)

    # Return the probabilities as a dictionary
    return {l: p.item() for l, p in zip(labels, probs[0])}