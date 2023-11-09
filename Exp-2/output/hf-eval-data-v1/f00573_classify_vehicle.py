from transformers import CLIPModel, CLIPProcessor
from PIL import Image


def classify_vehicle(image_path):
    """
    This function classifies images of vehicles including cars, motorcycles, trucks, and bicycles, based on their appearance.
    It uses the pre-trained model 'openai/clip-vit-base-patch32' from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    dict: A dictionary with the probabilities for each class.
    """
    # Load the pre-trained model and processor
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    # Load the image
    image = Image.open(image_path)
    
    # Process the image and text labels together for the model's input
    inputs = processor(text=['a car', 'a motorcycle', 'a truck', 'a bicycle'], images=image, return_tensors='pt', padding=True)
    
    # Get the model's outputs
    outputs = model(**inputs)
    
    # Get the probabilities for each class
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    return probs