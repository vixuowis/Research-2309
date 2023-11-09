from transformers import Swin2SRForConditionalGeneration
from PIL import Image


def upscale_image(low_res_image_path: str, high_res_image_path: str):
    """
    Function to upscale a low resolution image using the Swin2SRForConditionalGeneration model from Hugging Face Transformers.
    
    Parameters:
    low_res_image_path (str): Path to the low resolution image.
    high_res_image_path (str): Path to save the upscaled image.
    
    Returns:
    None
    """
    # Load the low-resolution image
    low_res_image = Image.open(low_res_image_path)
    
    # Load the pre-trained 'condef/Swin2SR-lightweight-x2-64' model
    model = Swin2SRForConditionalGeneration.from_pretrained('condef/Swin2SR-lightweight-x2-64')
    
    # Use the model to upscale the image
    high_res_image = model.upscale_image(low_res_image)
    
    # Save the upscaled image
    high_res_image.save(high_res_image_path)