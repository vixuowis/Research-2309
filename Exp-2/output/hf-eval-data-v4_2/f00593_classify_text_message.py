# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_text_message(text_message: str, candidate_labels: list) -> dict:
    """
    Classify a given text message into one of the predefined categories.

    Args:
        text_message (str): The text message to classify.
        candidate_labels (list): A list of strings representing the candidate
            categories for classification.

    Returns:
        dict: A dictionary containing the classification result and corresponding score.

    Raises:
        ValueError: If the candidate_labels list is empty.
    """
    if not candidate_labels:
        raise ValueError('Candidate labels must not be empty.')
    classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
    return classifier(text_message, candidate_labels)

# test_function_code --------------------

def test_classify_text_message():
    print('Testing started.')
    # Predefined labels to categorize the text messages
    labels = ['finances', 'health', 'entertainment']

    # Testing case 1 - Financial message
    print('Testing case [1/3] started.')
    financial_message = 'Your loan has been approved.'
    result_finance = classify_text_message(financial_message, labels)
    assert result_finance['labels'][0] == 'finances', f"Test case [1/3] failed: {result_finance}"

    # Testing case 2 - Health message
    print('Testing case [2/3] started.')
    health_message = 'Your doctor's appointment is scheduled for tomorrow.'
    result_health = classify_text_message(health_message, labels)
    assert result_health['labels'][0] == 'health', f"Test case [2/3] failed: {result_health}"

    # Testing case 3 - Entertainment message
    print('Testing case [3/3] started.')
    entertainment_message = 'New movie releases this weekend.'
    result_entertainment = classify_text_message(entertainment_message, labels)
    assert result_entertainment['labels'][0] == 'entertainment', f"Test case [3/3] failed: {result_entertainment}"
    print('Testing finished.')

# call_test_function_line --------------------

test_classify_text_message()