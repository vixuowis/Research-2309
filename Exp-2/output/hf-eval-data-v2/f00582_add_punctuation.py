# function_import --------------------

from transformers import pipeline

# function_code --------------------

def add_punctuation(user_message):
    """
    This function adds punctuation to a user's message using a token classification model.

    Args:
        user_message (str): The user's message that needs punctuation.

    Returns:
        str: The user's message with added punctuation.
    """
    punctuator = pipeline('token-classification', model='kredor/punctuate-all')
    corrected_user_message = punctuator(user_message)
    return corrected_user_message

# test_function_code --------------------

def test_add_punctuation():
    """
    This function tests the add_punctuation function by using a sample message.
    """
    sample_message = 'hello how are you'
    expected_output = 'Hello, how are you?'
    assert add_punctuation(sample_message) == expected_output

# call_test_function_code --------------------

test_add_punctuation()