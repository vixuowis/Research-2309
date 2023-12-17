# requirements_file --------------------

!pip install -U diffusers Pillow

# function_import --------------------

from diffusers import DDPMPipeline
from PIL import Image

# function_code --------------------

def generate_celebrity_image(model_id: str) -> Image:
    '''
    Generate a high-quality celebrity face image using a pretrained model.

    Args:
        model_id (str): The model ID of the pretrained Denoising Diffusion Probabilistic Model.

    Returns:
        Image: The generated celebrity face image as a PIL Image object.

    Raises:
        ValueError: If the model_id is not a string or is empty.
        RuntimeError: If the DDPM pipeline encounters an issue during image generation.
    '''
    if not isinstance(model_id, str) or not model_id:
        raise ValueError('The model_id must be a non-empty string.')
    try:
        ddpm_pipeline = DDPMPipeline.from_pretrained(model_id)
        generated_image = ddpm_pipeline().images[0]
        return generated_image
    except Exception as e:
        raise RuntimeError(f'Error occurred while generating the image: {e}')

# test_function_code --------------------

def test_generate_celebrity_image():
    print("Testing started.")

    # Test case 1: Valid model ID
    print("Testing case [1/3] started.")
    model_id = 'google/ddpm-ema-celebahq-256'
    generated_image = generate_celebrity_image(model_id)
    assert isinstance(generated_image, Image.Image), f"Test case [1/3] failed: The generated object is not an Image instance."

    # Test case 2: Invalid model ID (non-string)
    print("Testing case [2/3] started.")
    non_string_model_id = 123
    try:
        generate_celebrity_image(non_string_model_id)
        assert False, f"Test case [2/3] failed: Function did not raise a ValueError for non-string model_id."
    except ValueError:
        assert True

    # Test case 3: Empty model ID
    print("Testing case [3/3] started.")
    empty_model_id = ''
    try:
        generate_celebrity_image(empty_model_id)
        assert False, f"Test case [3/3] failed: Function did not raise a ValueError for empty model_id."
    except ValueError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_celebrity_image()