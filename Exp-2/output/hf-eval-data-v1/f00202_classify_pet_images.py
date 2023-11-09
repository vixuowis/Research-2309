from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Function to classify pet images
# Uses the pre-trained CLIP model from Hugging Face Transformers
# The function takes in the path to the image and returns the classification probabilities

def classify_pet_images(image_path):
    # Initialize the pre-trained CLIP model
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    # Initialize the CLIP processor
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    # Open the image
    image = Image.open(image_path)
    # Preprocess the input data
    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    # Pass the preprocessed inputs to the model
    outputs = model(**inputs)
    # Obtain the classification probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # Return the classification probabilities
    return probs