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
        Exception: If there is an error in generating the email.
    """
    try:
        text_generator = pipeline('text-generation', model='lewtun/tiny-random-mt5')
        generated_email = text_generator(seed_text, max_length=150)
        return generated_email
    except Exception as e:
        print(f'Error in generating email: {e}')
        raise

# test_function_code --------------------

def test_generate_welcome_email():
    """
    Test the generate_welcome_email function.
    """
    generated_email = generate_welcome_email()
    assert isinstance(generated_email, str), 'The generated email should be a string.'
    assert len(generated_email) <= 150, 'The length of the generated email should not exceed 150.'

# call_test_function_code --------------------

test_generate_welcome_email()