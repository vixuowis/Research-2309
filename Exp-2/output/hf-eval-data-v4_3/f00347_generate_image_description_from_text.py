# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_image_description_from_text(input_text: str) -> str:
    """
    Generate an image description from the given text.

    Args:
        input_text (str): The text input from which to generate the image description.

    Returns:
        str: The generated image description.

    Raises:
        ValueError: If the input text is not a string.
        RuntimeError: If the pipeline model fails to generate the image description.
    """
    if not isinstance(input_text, str):
        raise ValueError("Expected input text to be a string.")

    try:
        text_to_image = pipeline('text-to-image', model='prompthero/openjourney-v4')
        result = text_to_image(input_text)
        return result
    except Exception as e:
        raise RuntimeError("Failed to generate image description.") from e

# test_function_code --------------------

def test_generate_image_description_from_text():
    print("Testing started.")
    # Here we only use a simple string to simulate the input, as we cannot call the actual API.
    input_text = "A sunset over a mountain range"

    # Testing case 1: Valid string input
    print("Testing case [1/1] started.")
    try:
        description = generate_image_description_from_text(input_text)
        assert isinstance(description, str), "The output must be a string."
    except Exception as e:
        print("Test case [1/1] failed:", str(e))
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_description_from_text()