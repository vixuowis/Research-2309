# function_import --------------------

from transformers import pipeline

# function_code --------------------

def filter_inappropriate_messages(message_text):
    """
    Filters out inappropriate messages using a zero-shot classification model.

    Args:
        message_text (str): The message text to be classified.

    Returns:
        str: A warning message if the input message is inappropriate, otherwise a safe message notification.
    """
    classifier = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-3')
    message_classification = classifier(message_text, candidate_labels=['safe', 'inappropriate'])
    if message_classification['labels'][0] == 'inappropriate':
        return 'Warning: Inappropriate message detected.'
    else:
        return 'Safe message.'

# test_function_code --------------------

def test_filter_inappropriate_messages():
    assert filter_inappropriate_messages('Hello, how are you?') == 'Safe message.'
    assert filter_inappropriate_messages('You are stupid.') == 'Warning: Inappropriate message detected.'
    assert filter_inappropriate_messages('What is your favorite color?') == 'Safe message.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_filter_inappropriate_messages()