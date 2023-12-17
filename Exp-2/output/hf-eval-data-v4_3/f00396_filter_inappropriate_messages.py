# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def filter_inappropriate_messages(message_text):
    """
    Filters out any inappropriate messages using zero-shot classification.

    Args:
        message_text (str): The message text to classify and filter.

    Returns:
        str: A warning message if inappropriate content is detected; otherwise, returns 'Safe message'.

    Raises:
        ValueError: If the input message_text is not a string.
    """
    if not isinstance(message_text, str):
        raise ValueError('Input must be a string.')

    classifier = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-3')
    message_classification = classifier(message_text, candidate_labels=['safe', 'inappropriate'])
    if message_classification['labels'][0] == 'inappropriate':
        return 'Warning: Inappropriate message detected.'
    return 'Safe message.'

# test_function_code --------------------

def test_filter_inappropriate_messages():
    print('Testing started.')
    
    # Testing case 1: Safe message
    print('Testing case [1/2] started.')
    safe_message = 'How are you today?'
    assert filter_inappropriate_messages(safe_message) == 'Safe message.', \
    f'Test case [1/2] failed: expected Safe message.'
    
    # Testing case 2: Inappropriate message
    print('Testing case [2/2] started.')
    inappropriate_message = 'Offensive content...'
    assert filter_inappropriate_messages(inappropriate_message) == 'Warning: Inappropriate message detected.', \
    f'Test case [2/2] failed: expected Warning: Inappropriate message detected.'
    
    print('Testing finished.')

# call_test_function_line --------------------

test_filter_inappropriate_messages()