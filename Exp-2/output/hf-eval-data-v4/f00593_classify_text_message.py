# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_text_message(text_message, candidate_labels):
    """
    Classify a text message into categories such as 'finances', 'health', and 'entertainment'.

    Parameters:
    text_message (str): The text message to classify.
    candidate_labels (list): A list of strings representing the candidate categories.

    Returns:
    dict: The classification results, including labels and scores.
    """
    classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
    classification_result = classifier(text_message, candidate_labels)
    return classification_result

# test_function_code --------------------

def test_classify_text_message():
    print("Testing started.")
    # Test case 1: Classify a bank statement message
    text_message = 'Your monthly bank statement is now available.'
    candidate_labels = ['finances', 'health', 'entertainment']
    result = classify_text_message(text_message, candidate_labels)
    assert 'finances' in result['labels'], "Test case [1/2] failed: 'finances' label not detected."

    # Test case 2: Classify a health-related message
    text_message = 'Remember to schedule your annual physical exam.'
    result = classify_text_message(text_message, candidate_labels)
    assert 'health' in result['labels'], "Test case [2/2] failed: 'health' label not detected."
    print("Testing finished.")

# Run the test function
test_classify_text_message()