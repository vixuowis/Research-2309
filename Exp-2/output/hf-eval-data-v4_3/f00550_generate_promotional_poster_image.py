# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_promotional_poster_image(prompt, negative_prompt):
    """
    Generates a promotional poster image based on positive and negative textual prompts.

    Args:
        prompt (str): A description of the desired output image attributes.
        negative_prompt (str): A description of what to avoid in the output image.

    Returns:
        dict: The result from the text-to-image model pipeline.

    Raises:
        ValueError: If prompt is not provided.
    """
    if not prompt:
        raise ValueError('Prompt description must be provided.')
    model = pipeline('text-to-image', model='SG161222/Realistic_Vision_V1.4')
    return model(prompt, negative_prompt=negative_prompt)

# test_function_code --------------------

def test_generate_promotional_poster_image():
    print("Testing started.")
    positive_prompt = "A promotional poster ... vibrant colors."
    negative_prompt = "winter, snow, cloudy, low-resolution, dull colors, indoor, mountain"

    # Test case 1: Basic functionality with both prompts
    print("Testing case [1/1] started.")
    try:
        result = generate_promotional_poster_image(positive_prompt, negative_prompt)
        assert result is not None, "Test case [1/1] failed: No result returned."
    except Exception as e:
        assert False, f"Test case [1/1] failed with exception: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_promotional_poster_image()