# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def generate_church_image():
    """
    Use a pre-trained DDPM model to generate an image of a church and save it as a local file.

    Steps:
    1. Load the pre-trained DDPM model 'google/ddpm-ema-church-256'.
    2. Generate a church image using the DDPMPipeline.
    3. Save the generated image to a file named 'generated_church_image.png'.

    Returns:
    - The path to the local file containing the generated church image.
    """
    # Load pre-trained model
    model_id = 'google/ddpm-ema-church-256'
    ddpm_pipeline = DDPMPipeline.from_pretrained(model_id)
    
    # Generate church image
    generation = ddpm_pipeline()
    image = generation.images[0]
    
    # Save image to a local file
    file_path = 'generated_church_image.png'
    image.save(file_path)
    
    return file_path

# test_function_code --------------------

def test_generate_church_image():
    print("Testing started.")
    # Test the execution of the generate_church_image function
    file_path = generate_church_image()
    
    # Test case 1: Verify that the generated file exists
    print("Testing case [1/1] started.")
    try:
        with open(file_path, 'rb') as img_file:
            Image.open(img_file)
        print(f"Test case [1/1] passed: The image file '{file_path}' exists and is readable.")
    except (IOError, FileNotFoundError) as e:
        print(f"Test case [1/1] failed: {e}")

    print("Testing finished.")

# Run the test function
test_generate_church_image()