# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news_headlines(headline: str, candidate_labels: list) -> dict:
    """
    Classify news headlines into categories using a zero-shot classifier.

    Args:
        headline (str): The news headline to classify.
        candidate_labels (list): The list of categories to classify the headline into.

    Returns:
        dict: The classification results.

    Raises:
        ValueError: If the headline is not a string or candidate_labels is not a list.
    """

    # Check input types --------------------

    if not isinstance(headline, str):
        raise TypeError('the "headline" argument should be a string')

    if not isinstance(candidate_labels, list):
        raise TypeError('the "candidate_labels" argument should be a list of strings')

    # Classify the headline --------------------

    classifier = pipeline("zero-shot-classification")
    classification = classifier(headline, candidate_labels)
    return classification

# test_function_code --------------------

def test_classify_news_headlines():
    """Tests for the classify_news_headlines function"""
    headline1 = 'Apple just announced the newest iPhone X'
    candidate_labels1 = ['technology', 'sports', 'politics']
    result1 = classify_news_headlines(headline1, candidate_labels1)
    assert isinstance(result1, dict), 'Result must be a dictionary'
    assert 'labels' in result1, 'Result dictionary must contain labels'
    assert 'scores' in result1, 'Result dictionary must contain scores'

    headline2 = 'The Lakers won their game last night'
    candidate_labels2 = ['technology', 'sports', 'politics']
    result2 = classify_news_headlines(headline2, candidate_labels2)
    assert isinstance(result2, dict), 'Result must be a dictionary'
    assert 'labels' in result2, 'Result dictionary must contain labels'
    assert 'scores' in result2, 'Result dictionary must contain scores'

    print('All Tests Passed')


# call_test_function_code --------------------

test_classify_news_headlines()