# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def add_punctuation_to_message(user_message: str) -> str:
    """
    Add punctuation to a given user message using a pre-trained NLP model.

    Args:
        user_message (str): The text of the user message to punctuate.

    Returns:
        str: The punctuated user message.

    Raises:
        ValueError: If the user_message is empty.
    """
    if not user_message:
        raise ValueError('The user_message is empty.')

    punctuator = pipeline('token-classification', model='kredor/punctuate-all')
    corrected_message = punctuator(user_message)
    # Assuming the model's output can be directly used as corrected_message
    return corrected_message

# test_function_code --------------------

def test_add_punctuation_to_message():
    print("Testing started.")

    # Test case 1: Check punctuation for English message
    print("Testing case [1/2] started.")
    english_message = "hello how are you"
    punctuated_message = add_punctuation_to_message(english_message)
    assert punctuated_message.endswith('.'), f"Test case [1/2] failed: expected punctuation at the end."

    # Test case 2: Check ValueError for empty message
    print("Testing case [2/2] started.")
    try:
        add_punctuation_to_message('')
        assert False, "Test case [2/2] failed: ValueError expected for empty message."
    except ValueError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_add_punctuation_to_message()