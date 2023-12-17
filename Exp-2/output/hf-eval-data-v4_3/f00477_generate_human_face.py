# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DiffusionPipeline

# function_code --------------------

def generate_human_face(model_id: str) -> Image:
    """
    Generate a high-resolution image of a human face using the specified model.

    Args:
        model_id (str): The identifier of the pre-trained diffusion model.

    Returns:
        Image: An image object representing the generated human face.

    Raises:
        ValueError: If the model_id is empty or None.
    """
    if not model_id:
        raise ValueError('The model_id must be a non-empty string.')
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    image = pipeline()[0]
    return image


# test_function_code --------------------

def test_generate_human_face():
    print("Testing started.")

    # Test case 1: Valid model_id
    print("Testing case [1/3] started.")
    model_id = 'google/ncsnpp-celebahq-256'
    image = generate_human_face(model_id)
    assert image is not None, f"Test case [1/3] failed: Expected a valid Image object, got None."

    # Test case 2: Invalid model_id (empty string)
    print("Testing case [2/3] started.")
    invalid_model_id = ''
    try:
        image = generate_human_face(invalid_model_id)
        assert False, f"Test case [2/3] failed: ValueError expected but not raised."
    except ValueError as e:
        assert str(e) == 'The model_id must be a non-empty string.', f"Test case [2/3] failed with unexpected error message: {e}"

    print("Testing finished.")


# call_test_function_line --------------------

test_generate_human_face()