# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
from PIL import Image

# function_code --------------------

def sharpen_image(image_path):
    """
    This function sharpens an image using the Swin2SRForConditionalGeneration model from Hugging Face Transformers.
    
    Args:
        image_path (str): The path to the image to be sharpened.
    
    Returns:
        PIL.Image: The sharpened image.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Load the pre-trained model
    model = Swin2SRForConditionalGeneration.from_pretrained('condef/Swin2SR-lightweight-x2-64')
    
    # Process the image
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    
    # Return the sharpened image
    return outputs

# test_function_code --------------------

def test_sharpen_image():
    """
    This function tests the sharpen_image function by sharpening a test image and checking the output.
    """
    # Define the path to the test image
    test_image_path = 'test_image.jpg'
    
    # Sharpen the test image
    output = sharpen_image(test_image_path)
    
    # Check the output
    assert isinstance(output, Image.Image), 'Output should be a PIL Image.'

# call_test_function_code --------------------

test_sharpen_image()