# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_welcome_email(seed_text='Welcome to the company,'):
    """
    Generate a welcome email for a new employee joining the company.

    Args:
        seed_text (str): The seed text to start the email. Default is 'Welcome to the company,'.

    Returns:
        str: The generated welcome email.

    Raises:
        TypeError: If the model is not compatible with .generate().
    """
    text_generator = pipeline('text-generation', model='lewtun/tiny-random-mt5')
    try:
        generated_email = text_generator(seed_text, max_length=150)
    except TypeError as e:
        print(f'Error: {e}')
        return None
    return generated_email

# test_function_code --------------------

def test_generate_welcome_email():
    """
    Test the generate_welcome_email function.
    """
    # Test with default seed text
    result = generate_welcome_email()
    assert isinstance(result, str), 'The result should be a string.'

    # Test with custom seed text
    result = generate_welcome_email('Hello,')
    assert isinstance(result, str), 'The result should be a string.'

    # Test with incompatible model
    result = generate_welcome_email('Hello,')
    assert result is None, 'The result should be None when the model is incompatible.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_welcome_email()