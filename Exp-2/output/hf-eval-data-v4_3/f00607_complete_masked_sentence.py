# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_masked_sentence(template: str) -> str:
    """
    Completes a sentence by filling in a masked token using BERT large cased model.

    Args:
        template: A string containing a sentence with the '[MASK]' token to be filled.

    Returns:
        A string representing the completed sentence with the '[MASK]' token filled.

    Raises:
        ValueError: If the input sentence does not contain any '[MASK]' token.
    """
    if '[MASK]' not in template:
        raise ValueError('Input sentence must contain a [MASK] token.')

    unmasker = pipeline('fill-mask', model='bert-large-cased')
    result = unmasker(template)
    return result[0]['sequence']

# test_function_code --------------------

def test_complete_masked_sentence():
    print('Testing started.')
    template = 'Hello, I\'m a [MASK]...'

    # Test case 1: Check if it completes the sentence correctly
    print('Testing case [1/1] started.')
    completed_sentence = complete_masked_sentence(template)
    assert '[MASK]' not in completed_sentence, f'Test case [1/1] failed: Mask not filled. Output = {completed_sentence}'
    print('Testing finished.')

# call_test_function_line --------------------

test_complete_masked_sentence()