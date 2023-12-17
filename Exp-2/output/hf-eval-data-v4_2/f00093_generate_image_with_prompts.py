# requirements_file --------------------

pip install -U torch PIL diffusers

# function_import --------------------

import torch
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

# function_code --------------------

def generate_image_with_prompts(prompt: str, negative_prompt: str, strength: float = 0.7, initial_image: Image.Image = None) -> Image.Image:
    """
    Generate an image based on a textual description. Optionally, an initial image can be provided.

    Args:
        prompt (str): The text prompt based on which the image is generated.
        negative_prompt (str): The negative text prompt to avoid certain features in the generated image.
        strength (float): The strength of the prompt effect on the generated image (default is 0.7).
        initial_image (Image.Image): The initial image to modify (optional).

    Returns:
        Image.Image: The generated image based on the provided text prompts.

    Raises:
        ValueError: If both prompt and negative_prompt are empty.
    """
    if not prompt and not negative_prompt:
        raise ValueError('Prompt and negative prompt cannot both be empty.')

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth',
        torch_dtype=torch.float16,
    ).to('cuda')

    if initial_image:
        input_image = {'image': initial_image}
    else:
        input_image = {}

    return pipe(prompt=prompt, negative_prompt=negative_prompt, strength=strength, **input_image).images[0]

# test_function_code --------------------

def test_generate_image_with_prompts():
    print("Testing started.")
    # Since the actual Stable Diffusion model requires a GPU environment and large dataset,
    # we would mock the behavior here assuming a perfect scenario where the image is generated successfully.

    # Test case 1: Only positive prompt
    print("Testing case [1/2] started.")
    try:
        img = generate_image_with_prompts('two tigers', 'bad, deformed, ugly, bad anatomy', strength=0.7, initial_image=None)
        assert img is not None, "Test case [1/2] failed: Image not generated."
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Test case 2: Raise ValueError when prompts are empty
    print("Testing case [2/2] started.")
    try:
        img = generate_image_with_prompts('', '', strength=0.7, initial_image=None)
        assert False, "Test case [2/2] failed: ValueError not raised."
    except ValueError:
        pass  # Expected
    except Exception as e:
        assert False, f"Test case [2/2] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_with_prompts()