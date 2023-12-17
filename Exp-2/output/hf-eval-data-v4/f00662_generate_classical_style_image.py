# requirements_file --------------------

!pip install -U diffusers requests pillow

# function_import --------------------

from diffusers import DDPMPipeline
from PIL import Image
import requests

# function_code --------------------

def generate_classical_style_image(output_path):
    """
    Generate an image with a classical painting style using a pretrained diffusion model and save it as a PNG file.

    Parameters:
    output_path (str): The file path to save the generated image as a PNG file.
    """
    # Load the pretrained diffusion model from Hugging Face Transformers
    pipeline = DDPMPipeline.from_pretrained('johnowhitaker/sd-class-wikiart-from-bedrooms')

    # Generate the image using the pipeline
    image = pipeline().images[0]

    # Convert to PIL Image to save as PNG
    pil_image = Image.fromarray(image.numpy())

    # Save the image as a PNG file
    pil_image.save(output_path, 'PNG')
    print(f"Image saved at {output_path}")

# test_function_code --------------------

import os

def test_generate_classical_style_image():
    print("Testing started.")
    
    output_path = "test_classical_image.png"
    # Run the image generation function
    generate_classical_style_image(output_path)
    
    # Check if the image file was created
    assert os.path.exists(output_path), f"Test case failed: Image file not found at {output_path}"
    
    # Optionally: Check the size of the generated file to ensure it's not empty
    assert os.path.getsize(output_path) > 0, "Test case failed: The image file is empty."
    
    # Clean up the generated test image file after testing
    if os.path.exists(output_path):
        os.remove(output_path)
    
    print("Testing finished successfully.")

test_generate_classical_style_image()