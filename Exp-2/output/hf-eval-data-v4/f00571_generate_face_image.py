# requirements_file --------------------

!pip install -U diffusers requests pillow

# function_import --------------------

from diffusers import DiffusionPipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def generate_face_image(model_id='CompVis/ldm-celebahq-256', num_inference_steps=200):
    # Instantiate the pipeline for generating images
    pipeline = DiffusionPipeline.from_pretrained(model_id)

    # Generate an image using the specified number of inference steps
    image = pipeline(num_inference_steps=num_inference_steps)[0]

    # Save the generated image to a BytesIO object and return it
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    return Image.open(image_bytes)

# test_function_code --------------------

def test_generate_face_image():
    print("Testing generate_face_image function.")

    # Test case 1: Generate an image with default parameters
    print("Generating image with default parameters.")
    image = generate_face_image()
    assert image is not None, "Failed to generate image with default parameters."

    # Test case 2: Generate an image with 100 inference steps
    print("Generating image with 100 inference steps.")
    image = generate_face_image(num_inference_steps=100)
    assert image is not None, "Failed to generate image with 100 inference steps."

    # Test case 3: Generate an image with a non-default model
    print("Generating image with a non-default model.")
    custom_model_id = 'CompVis/ldm-stable-diffusion-v1-4'
    image = generate_face_image(model_id=custom_model_id)
    assert image is not None, "Failed to generate image with a non-default model."

    print("All tests passed.")

test_generate_face_image()