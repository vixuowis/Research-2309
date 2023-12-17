# requirements_file --------------------

!pip install -U torch diffusers PIL

# function_import --------------------

import torch
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

# function_code --------------------

def generate_promotional_image(prompt, negative_prompt=None, init_image=None, strength=0.7):
    """Generate a promotional image based on the text prompt using Stable Diffusion model.

    :param prompt: Text prompt to generate the image.
    :param negative_prompt: (Optional) Negative text prompt to avoid certain features.
    :param init_image: (Optional) Initial image to start from.
    :param strength: (Optional) Strength of the prompt effect on the generated image.
    :return: An image generated based on the specified prompt.
    """
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth',
        torch_dtype=torch.float16,
    ).to('cuda')

    # Generate the image
    result = pipe(prompt=prompt, image=init_image, negative_prompt=negative_prompt, strength=strength)
    image = result.images[0]

    return image

# test_function_code --------------------

def test_generate_promotional_image():
    print("Testing generate_promotional_image function.")

    # Test case 1: Generate image without init_image
    print("Testing case [1/3] started.")
    promo_image = generate_promotional_image("two tigers")
    assert isinstance(promo_image, Image.Image), "Test case [1/3] failed: Generated object is not an image."

    # Test case 2: Generate image with negative prompts
    print("Testing case [2/3] started.")
    promo_image = generate_promotional_image("two tigers", negative_prompt="bad, deformed, ugly")
    assert isinstance(promo_image, Image.Image), "Test case [2/3] failed: Generated object is not an image."

    # Test case 3: Generate image at different strength
    print("Testing case [3/3] started.")
    promo_image = generate_promotional_image("two tigers", strength=0.5)
    assert isinstance(promo_image, Image.Image), "Test case [3/3] failed: Generated object is not an image."

    print("Testing finished.")

# Run the test function
test_generate_promotional_image()