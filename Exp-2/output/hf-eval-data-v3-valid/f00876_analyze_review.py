# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_review(review_text: str) -> dict:
    """
    Analyze the sentiment of a movie review using zero-shot classification.

    Args:
        review_text (str): The text of the movie review.

    Returns:
        dict: The result of the zero-shot classification, including the labels and scores.
    """
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    result = nlp(review_text, ['positive', 'negative'])
    return result

# test_function_code --------------------

def test_analyze_review():
    """
    Test the analyze_review function.
    """
    review_positive = 'The movie was great!'
    review_negative = 'The movie was terrible!'
    result_positive = analyze_review(review_positive)
    result_negative = analyze_review(review_negative)
    assert result_positive['labels'][0] == 'positive', 'Test Case 1 Failed'
    assert result_negative['labels'][0] == 'negative', 'Test Case 2 Failed'
    print('All Tests Passed')

# call_test_function_code --------------------

test_analyze_review()