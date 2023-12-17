# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def filter_inappropriate_messages(message_text):
    """
    Filters out inappropriate messages using zero-shot classification.

    Args:
        message_text (str): The text of the message to classify.

    Returns:
        str: 'safe' if the message is appropriate, 'inappropriate' if not.
    """
    classifier = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-3')
    message_classification = classifier(message_text, candidate_labels=['safe', 'inappropriate'])
    if message_classification['labels'][0] == 'inappropriate':
        return 'inappropriate'
    return 'safe'

# test_function_code --------------------

def test_filter_inappropriate_messages():
    print("Testing filter_inappropriate_messages function.")

    # Test case 1: Safe message
    safe_message = "Hello, how are you today?"
    assert filter_inappropriate_messages(safe_message) == 'safe', "Test case 1 failed: Safe message was marked as inappropriate."

    # Test case 2: Inappropriate message
    inappropriate_message = "I hate you!"
    assert filter_inappropriate_messages(inappropriate_message) == 'inappropriate', "Test case 2 failed: Inappropriate message was not detected."

    # Test case 3: Ambiguous message
    ambiguous_message = "That's a weird thing to say."
    assert filter_inappropriate_messages(ambiguous_message) == 'safe', "Test case 3 failed: Ambiguous message was marked as inappropriate."

    print("All tests passed.")

    test_filter_inappropriate_messages()