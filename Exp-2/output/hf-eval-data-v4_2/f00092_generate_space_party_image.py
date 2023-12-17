# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_space_party_image(prompt: str) -> str:
    """
    Generate an image of a space party based on the given text prompt.

    Args:
        prompt (str): A text prompt describing the space party.

    Returns:
        str: The filepath to the saved image.

    Raises:
        RuntimeError: If the model fails to generate the image.
    """
    model_id = 'stabilityai/stable-diffusion-2-1'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    image = pipe(prompt).images[0]
    file_path = 'space_party.png'
    image.save(file_path)
    return file_path

# test_function_code --------------------

def test_generate_space_party_image():
    print("Testing started.")
    test_prompt = "A space party with astronauts and aliens having fun together"

    print("Testing case [1/1] started.")
    file_path = generate_space_party_image(test_prompt)
    assert file_path.endswith('.png'), "Test case [1/1] failed: The function should return a filepath with a .png extension"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_space_party_image()