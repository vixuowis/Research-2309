# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_headline(headline: str, candidate_labels: list = ['technology', 'sports', 'politics']) -> dict:
    """
    Classify a news headline into one of the given categories using a zero-shot classifier.

    Args:
        headline (str): The news headline to classify.
        candidate_labels (list, optional): A list of categories to classify the headline into. Defaults to ['technology', 'sports', 'politics'].

    Returns:
        dict: The predicted category and its corresponding score.
    """
    headlines_classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    headline_category = headlines_classifier(headline, candidate_labels)
    return headline_category

# test_function_code --------------------

def test_classify_headline():
    """
    Test the classify_headline function.
    """
    headline = 'Apple just announced the newest iPhone X'
    expected_output = {'labels': ['technology', 'sports', 'politics'], 'scores': [0.9883, 0.0074, 0.0043], 'sequence': 'Apple just announced the newest iPhone X'}
    assert classify_headline(headline)['labels'][0] == expected_output['labels'][0]

# call_test_function_code --------------------

test_classify_headline()