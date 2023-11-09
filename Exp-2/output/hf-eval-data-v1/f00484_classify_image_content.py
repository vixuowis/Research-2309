from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

# Function to classify an image's content and check if it contains a cat or a dog
# Uses the pretrained CLIP model 'openai/clip-vit-base-patch16' from Hugging Face Transformers
# The function takes an image URL as input and returns the probabilities of the image containing a cat or a dog

def classify_image_content(image_url):
    # Load the pretrained CLIP model and the corresponding CLIPProcessor
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    # Open the image using the PIL library
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the image with the CLIPProcessor to create the required inputs for the model
    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)

    # Pass the preprocessed inputs to the model and get the output logits_per_image
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image

    # Transform the logits_per_image into probabilities using the softmax function
    probs = logits_per_image.softmax(dim=1)

    # Return the probabilities
    return probs.tolist()