from transformers import Swin2SRForImageSuperResolution
from PIL import Image


def upscale_image(image_path: str, model_path: str = 'caidas/swin2sr-classical-sr-x2-64') -> None:
    """
    Function to upscale an image using a pre-trained model from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image to be upscaled.
    model_path (str): The path to the pre-trained model. Default is 'caidas/swin2sr-classical-sr-x2-64'.
    
    Returns:
    None. The function saves the upscaled image in the same directory as the original image.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Load the pre-trained model
    model = Swin2SRForImageSuperResolution.from_pretrained(model_path)
    
    # Upscale the image
    upscaled_image = model(image)
    
    # Save the upscaled image
    upscaled_image.save('upscaled_' + image_path)