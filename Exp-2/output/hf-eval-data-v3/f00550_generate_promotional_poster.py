# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_promotional_poster(prompt: str, negative_prompt: str):
    """
    Generate a promotional poster for a new line of summer clothing.

    Args:
        prompt (str): A textual description of the desired image.
        negative_prompt (str): A textual description of what should not be in the image.

    Returns:
        dict: The generated image.

    Raises:
        OSError: If the model 'SG161222/Realistic_Vision_V1.4' is not found.
    """
    try:
        model = pipeline('text-to-image', model='SG161222/Realistic_Vision_V1.4')
        result = model(prompt, negative_prompt=negative_prompt)
        return result
    except Exception as e:
        raise OSError('Model not found. Please check the model name.') from e

# test_function_code --------------------

def test_generate_promotional_poster():
    """
    Test the function generate_promotional_poster.
    """
    prompt = 'A promotional poster for a new line of summer clothing featuring happy people wearing the clothes, with a sunny beach background, clear blue sky, and palm trees. Image dimensions should be poster-sized, high-resolution, and vibrant colors.'
    negative_prompt = 'winter, snow, cloudy, low-resolution, dull colors, indoor, mountain'
    try:
        result = generate_promotional_poster(prompt, negative_prompt)
        assert isinstance(result, dict), 'The result should be a dictionary.'
        print('Test passed.')
    except Exception as e:
        print('Test failed. Error: ', e)

# call_test_function_code --------------------

test_generate_promotional_poster()