# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_text(text_message, candidate_labels):
    """
    Classify a given text message into one of the provided categories.

    Args:
        text_message (str): The text message to be classified.
        candidate_labels (list): A list of strings representing the candidate categories.

    Returns:
        dict: A dictionary containing the 'labels' and 'scores'. 'labels' is a list of the candidate labels in descending order of score. 'scores' is a list of the corresponding scores.
    """
    classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
    classification_result = classifier(text_message, candidate_labels)
    return classification_result

# test_function_code --------------------

def test_classify_text():
    """
    Test the classify_text function with a sample text message and candidate labels.
    """
    text_message = 'Your monthly bank statement is now available.'
    candidate_labels = ['finances', 'health', 'entertainment']
    classification_result = classify_text(text_message, candidate_labels)
    assert isinstance(classification_result, dict)
    assert 'labels' in classification_result
    assert 'scores' in classification_result
    assert isinstance(classification_result['labels'], list)
    assert isinstance(classification_result['scores'], list)

# call_test_function_code --------------------

test_classify_text()