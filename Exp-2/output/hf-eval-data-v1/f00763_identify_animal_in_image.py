from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch


def identify_animal_in_image(url):
    """
    This function identifies whether an image contains a cat or a dog using a pre-trained ChineseCLIPModel.

    Args:
        url (str): The URL of the image to be classified.

    Returns:
        str: The identified animal ('猫' for cat, '狗' for dog).
    """
    # Load the pre-trained ChineseCLIPModel and ChineseCLIPProcessor
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')

    # Open the image from the provided URL
    image = Image.open(requests.get(url, stream=True).raw)

    # Define the Chinese texts for cat and dog
    texts = ['猫', '狗']

    # Process the image and text inputs
    inputs = processor(images=image, text=texts, return_tensors='pt', padding=True)

    # Calculate image and text features
    outputs = model(**inputs)

    # Normalize the features and compute the probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Determine the category with the highest probability
    highest_prob_idx = probs.argmax(dim=1)
    animal = texts[highest_prob_idx]

    return animal