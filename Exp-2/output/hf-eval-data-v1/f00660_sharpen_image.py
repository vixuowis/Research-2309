from transformers import Swin2SRForConditionalGeneration
from PIL import Image


def sharpen_image(image_path):
    """
    This function sharpens the image captured from the drone in real-time using the Swin2SRForConditionalGeneration model.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    PIL.Image: The sharpened image.
    """
    # Load the image from the provided path
    image = Image.open(image_path)
    
    # Load the pre-trained model
    model = Swin2SRForConditionalGeneration.from_pretrained('condef/Swin2SR-lightweight-x2-64')
    
    # Prepare the inputs for the model
    inputs = feature_extractor(images=image, return_tensors='pt')
    
    # Use the model to sharpen the image
    outputs = model(**inputs)
    
    # Return the sharpened image
    return outputs