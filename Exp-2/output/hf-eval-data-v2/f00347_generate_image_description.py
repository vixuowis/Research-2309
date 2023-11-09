# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_image_description(input_text: str) -> str:
    """
    Generate an image description from the given text using the 'prompthero/openjourney-v4' model.

    Args:
        input_text (str): The text input to generate the image description from.

    Returns:
        str: The generated image description.
    """
    text_to_image = pipeline('text-to-image', model='prompthero/openjourney-v4')
    result = text_to_image(input_text)
    return result

# test_function_code --------------------

def test_generate_image_description():
    """
    Test the 'generate_image_description' function.
    """
    input_text = 'A beautiful sunset over the ocean.'
    result = generate_image_description(input_text)
    assert isinstance(result, str), 'The result should be a string.'
    assert len(result) > 0, 'The result should not be an empty string.'

# call_test_function_code --------------------

test_generate_image_description()