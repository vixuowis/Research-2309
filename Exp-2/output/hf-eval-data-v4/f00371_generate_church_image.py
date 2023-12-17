# requirements_file --------------------

!pip install -U diffusers requests Pillow

# function_import --------------------

from diffusers import DDPMPipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------


def generate_church_image():
    """
    Generate a high-quality image of a church using unconditional image generation.

    Returns:
        image: An Image object containing the generated church image.
    """
    # Load the pre-trained DDPM model
    model_id = 'google/ddpm-church-256'
    ddpm = DDPMPipeline.from_pretrained(model_id)

    # Generate the image
    generated_image = ddpm().images[0]

    return generated_image

# test_function_code --------------------


def test_generate_church_image():
    print("Testing function generate_church_image.")

    # Generate the church image
    generated_image = generate_church_image()

    # The result should be an instance of Image.Image from PIL
    assert isinstance(generated_image, Image.Image), "The generated image is not a PIL Image instance."

    # Save the image to verify manually if needed
    generated_image.save('test_generated_church_image.png')
    print("Test passed - The function generate_church_image generated an image successfully.")

# Run the test
if __name__ == "__main__":
    test_generate_church_image()