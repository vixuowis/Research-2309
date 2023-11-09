from transformers import pipeline
from PIL import Image


def detect_ai_generated_anime(image_path):
    """
    This function takes an image path as input and uses a pre-trained model to classify whether the provided anime art is created by a human or generated through AI.
    
    Parameters:
    image_path (str): The path to the image file
    
    Returns:
    str: The classification result
    """
    # Load the image from the provided path
    image = Image.open(image_path)
    
    # Create an image classification model with the pre-trained model 'saltacc/anime-ai-detect'
    anime_detector = pipeline('image-classification', model='saltacc/anime-ai-detect')
    
    # Pass the image to the image classification model
    classification_result = anime_detector(image)
    
    # Return the classification result
    return classification_result