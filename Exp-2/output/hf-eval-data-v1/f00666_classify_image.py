from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Function to classify images into categories
# @param image_path: Path to the image that needs to be classified
# @return: Probabilities of the image belonging to each category

def classify_image(image_path):
    # Load the pre-trained CLIP model
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    # Load the image data
    image = Image.open(image_path)

    # Define the list of categories
    categories = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

    # Process the image and the categories
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)

    # Pass them to the CLIP model and extract the logits_per_image
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image

    # Calculate the probabilities
    probs = logits_per_image.softmax(dim=1)

    return probs.tolist()