# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_fill_in_the_blank(sentence, keyword):
    """
    Generates a fill-in-the-blank question by masking a keyword in a given sentence.

    Args:
        sentence (str): The sentence where the keyword will be masked.
        keyword (str): The keyword to mask in the sentence.

    Returns:
        str: A masked sentence with the keyword replaced by '[MASK]'.

    Raises:
        ValueError: If the keyword is not found in the sentence.
    """
    if keyword not in sentence:
        raise ValueError('Keyword not found in the sentence.')
    return sentence.replace(keyword, '[MASK]')

# test_function_code --------------------

def test_generate_fill_in_the_blank():
    print('Testing started.')

    # Case 1: Normal sentence
    print('Testing case [1/3] started.')
    sentence = 'Learning languages is fun.'
    keyword = 'languages'
    expected_result = 'Learning [MASK] is fun.'
    assert generate_fill_in_the_blank(sentence, keyword) == expected_result, f'Test case [1/3] failed: Expected {expected_result}'

    # Case 2: Keyword at the start
    print('Testing case [2/3] started.')
    sentence = 'Languages is the key to communication.'
    keyword = 'Languages'
    expected_result = '[MASK] is the key to communication.'
    assert generate_fill_in_the_blank(sentence, keyword) == expected_result, f'Test case [2/3] failed: Expected {expected_result}'

    # Case 3: Keyword not in sentence
    print('Testing case [3/3] started.')
    sentence = 'Speaking multiple languages is impressive.'
    keyword = 'music'
    try:
        generate_fill_in_the_blank(sentence, keyword)
        assert False, 'Test case [3/3] failed: ValueError not raised.'
    except ValueError as e:
        assert str(e) == 'Keyword not found in the sentence.', f'Test case [3/3] failed: Unexpected exception message {str(e)}'

    print('Testing finished.')

# call_test_function_line --------------------

test_generate_fill_in_the_blank()