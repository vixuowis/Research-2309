# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_masked_text(masked_text):
    """
    Identify the masked words in the provided text using a masked language model.

    Args:
        masked_text (str): Text with one or more '<mask>' tokens that
            need to be filled with predicted words.

    Returns:
        list: A list of dictionaries where each dictionary represents
            the top prediction for each '<mask>' token in the text.

    Raises:
        ValueError: If the 'masked_text' is not a string or if it
            doesn't contain any '<mask>' token.
    """
    # Validating the input text
    if not isinstance(masked_text, str) or '<mask>' not in masked_text:
        raise ValueError('Invalid input text. It must be a string containing \'<mask>\' tokens.')
    
    # Initializing the fill-mask pipeline
    mask_unmasker = pipeline('fill-mask', model='xlm-roberta-large')
    
    # Getting predictions for the masked tokens
    return mask_unmasker(masked_text)

# test_function_code --------------------

def test_fill_masked_text():
    print("Testing started.")

    # Test case 1: Valid input with one mask
    print("Testing case [1/2] started.")
    result1 = fill_masked_text("Alligators are <mask> animals.")
    assert len(result1) == 1 and result1[0]['sequence'].strip() != "", "Test case [1/2] failed: Expected at least one prediction result."

    # Test case 2: Invalid input without any mask
    print("Testing case [2/2] started.")
    try:
        fill_masked_text("Alligators are dangerous animals.")
        assert False, "Test case [2/2] failed: ValueError expected for input without '<mask>' token."
    except ValueError as e:
        assert str(e) == 'Invalid input text. It must be a string containing \'<mask>\' tokens.', "Test case [2/2] failed: Incorrect ValueError message."
    
    print("Testing finished.")

# call_test_function_line --------------------

test_fill_masked_text()