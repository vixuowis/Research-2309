from transformers import pipeline
import torch

# Function to transform an image using the 'GreeneryScenery/SheepsControlV5' model from Hugging Face
# @param input_image_path: The path to the image file to be transformed
# @return: The transformed image

def transform_image(input_image_path):
    # Create an Image-to-Image model
    image_transformer = pipeline('image-to-image', model='GreeneryScenery/SheepsControlV5')
    # Process the input image with the model
    stylized_image = image_transformer(input_image_path)
    return stylized_image