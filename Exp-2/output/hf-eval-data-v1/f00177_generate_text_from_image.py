from transformers import pipeline
import os


def generate_text_from_image(image_path):
    """
    This function takes an image path as input and generates a text description based on the content of the image.
    It uses the pre-trained model 'microsoft/git-large-r-textcaps' from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    str: The generated text description of the image.
    """
    # Check if the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")
    
    # Create a text generation pipeline with the pre-trained model
    img2text_pipeline = pipeline('text-generation', model='microsoft/git-large-r-textcaps')
    
    # Load the image data
    with open(image_path, 'rb') as f:
        image = f.read()
    
    # Generate a text description based on the content of the image
    text_output = img2text_pipeline(image)[0]['generated_text']
    
    return text_output