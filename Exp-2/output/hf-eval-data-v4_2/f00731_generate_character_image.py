# requirements_file --------------------

!pip install -U diffusers torch

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_character_image(prompt, negative_prompt):
    """
    Generates an anime character image based on the provided prompt and negative prompt.

    Args:
        prompt (str): A description of the desired character features.
        negative_prompt (str): A description of features to exclude from the image.

    Returns:
        Image: The generated anime character image.

    Raises:
        ValueError: If the prompt or negative_prompt are empty.

    """
    if not prompt or not negative_prompt:
        raise ValueError('The prompt and negative_prompt cannot be empty.')

    model_id = 'dreamlike-art/dreamlike-anime-1.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]

    return image

# test_function_code --------------------

def test_generate_character_image():
    print("Testing started.")
    prompt = 'anime, high quality, 1girl, solo, long hair, looking at viewer, blush, smile'
    negative_prompt = 'low quality, worst quality, blurry'

    # Test case 1: Valid prompt and negative_prompt
    print("Testing case [1/3] started.")
    image = generate_character_image(prompt, negative_prompt)
    assert image is not None, "Test case [1/3] failed: The function should return an image."

    # Test case 2: Empty prompt
    print("Testing case [2/3] started.")
    try:
        generate_character_image('', negative_prompt)
        assert False, "Test case [2/3] failed: The function should raise a ValueError for empty prompt."
    except ValueError:
        assert True

    # Test case 3: Empty negative_prompt
    print("Testing case [3/3] started.")
    try:
        generate_character_image(prompt, '')
        assert False, "Test case [3/3] failed: The function should raise a ValueError for empty negative_prompt."
    except ValueError:
        assert True
    
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_character_image()