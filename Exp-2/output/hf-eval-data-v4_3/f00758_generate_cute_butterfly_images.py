# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cute_butterfly_images(model_name: str):
    """
    Generate images of cute butterflies using a specified diffusion model.

    Args:
        model_name (str): The name of the diffusion model to use.

    Returns:
        List[PIL.Image.Image]: A list containing the generated butterfly image(s).

    Raises:
        ValueError: If the model_name is not specified or empty.
        RuntimeError: If the model cannot be loaded or generation fails.
    """
    if not model_name:
        raise ValueError('Model name must be specified.')
    try:
        pipeline = DDPMPipeline.from_pretrained(model_name)
        generated_data = pipeline()
        images = generated_data.images
        return images
    except Exception as e:
        raise RuntimeError(f'Failed to generate images: {e}')

# test_function_code --------------------

def test_generate_cute_butterfly_images():
    print("Testing started.")

    # Test case 1: Check if providing a valid model name returns a list of images
    print("Testing case [1/2] started.")
    images = generate_cute_butterfly_images('myunus1/diffmodels_galaxies_scratchbook')
    assert isinstance(images, list) and images, f"Test case [1/2] failed: Expected a list of images, got {type(images)} or empty list."

    # Test case 2: Check if providing an invalid or empty model name raises ValueError
    print("Testing case [2/2] started.")
    try:
        generate_cute_butterfly_images('')
    except ValueError as ve:
        assert str(ve) == 'Model name must be specified.', f"Test case [2/2] failed: {ve}"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_cute_butterfly_images()