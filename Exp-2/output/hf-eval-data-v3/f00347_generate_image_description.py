# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_image_description(input_text: str) -> str:
    """
    Generate an image description based on the given text input.

    Args:
        input_text (str): The text input to generate an image description.

    Returns:
        str: The generated image description.

    Raises:
        OSError: If the model 'prompthero/openjourney-v4' is not found.
    """
    try:
        text_to_image = pipeline('text-to-image', model='prompthero/openjourney-v4')
        result = text_to_image(input_text)
        return result
    except Exception as e:
        raise OSError('Model not found. Please check the model name.') from e

# test_function_code --------------------

def test_generate_image_description():
    """
    Test the function 'generate_image_description'.
    """
    try:
        # Test case 1: Normal case
        input_text = 'A beautiful sunset over the ocean.'
        result = generate_image_description(input_text)
        assert isinstance(result, str), 'The result should be a string.'

        # Test case 2: Empty string
        input_text = ''
        result = generate_image_description(input_text)
        assert isinstance(result, str), 'The result should be a string.'

        # Test case 3: Long string
        input_text = 'A' * 1000
        result = generate_image_description(input_text)
        assert isinstance(result, str), 'The result should be a string.'

        print('All Tests Passed')
    except Exception as e:
        print(f'Test failed. Error: {str(e)}')

# call_test_function_code --------------------

test_generate_image_description()