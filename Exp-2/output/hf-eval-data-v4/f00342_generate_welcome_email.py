# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_welcome_email(seed_text='Welcome to the company,'):
    """
    Generate a welcome email for a new employee using a text generation model.

    Parameters:
        seed_text (str): The initial text to start generating the email.

    Returns:
        str: The generated welcome.email.
    """
    text_generator = pipeline('text-generation', model='lewtun/tiny-random-mt5')
    generated_email = text_generator(seed_text, max_length=150)[0]['generated_text']
    return generated_email

# test_function_code --------------------

def test_generate_welcome_email():
    print("Testing started.")

    # Test case 1: Seed text is provided
    print("Testing case [1/1] started.")
    seed_text = 'Welcome to the company, John!'
    email = generate_welcome_email(seed_text)
    assert email.startswith(seed_text), f"Test case [1/1] failed: Generated email does not start with seed text."
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_welcome_email()