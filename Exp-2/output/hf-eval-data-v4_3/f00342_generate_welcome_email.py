# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_welcome_email(seed_text):
    """
    Generate a welcome email using a text-generation pipeline.

    Args:
        seed_text (str): The seed text to initiate the text-generation process.

    Returns:
        str: The generated welcome email text.

    Raises:
        ValueError: If the seed_text is empty.
    """
    if not seed_text:
        raise ValueError('Seed text must not be empty.')
    text_generator = pipeline('text-generation', model='lewtun/tiny-random-mt5')
    result = text_generator(seed_text, max_length=150)
    return result[0]['generated_text']

# test_function_code --------------------

def test_generate_welcome_email():
    print("Testing started.")
    # Test case 1: Seed text is provided
    print("Testing case [1/2] started.")
    seed_text = 'Welcome to the company, John.'
    generated_text = generate_welcome_email(seed_text)
    assert generated_text.startswith(seed_text), f"Test case [1/2] failed: Generated text does not start with the provided seed text."

    # Test case 2: Seed text is empty
    print("Testing case [2/2] started.")
    seed_text_empty = ''
    try:
        generate_welcome_email(seed_text_empty)
        assert False, 'Test case [2/2] failed: ValueError was not raised for empty seed text.'
    except ValueError as e:
        assert str(e) == 'Seed text must not be empty.', f"Test case [2/2] failed: ValueError message does not match expected."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_welcome_email()