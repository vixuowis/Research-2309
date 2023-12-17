# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors torch

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_space_party_image(prompt='a space party with astronauts and aliens having fun together'):
    """
    Generate an image based on a prompt describing a space party with astronauts and aliens.

    :param prompt: Text prompt to generate the image based on.
    :return: Generated image object.
    """
    # Load the pre-trained Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', torch_dtype=torch.float16)
    pipe.to('cuda')

    # Generate an image from the prompt
    image = pipe(prompt).images[0]
    
    return image

# test_function_code --------------------

def test_generate_space_party_image():
    print("Testing started.")

    # Test case 1: Generate image with default prompt
    print("Testing case [1/1] started.")
    image = generate_space_party_image()
    assert image is not None, f"Test case [1/1] failed: Expected an image, got None"
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_space_party_image()