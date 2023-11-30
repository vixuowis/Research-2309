# function_import --------------------

from transformers import pipeline

# function_code --------------------

def news_category_detection(text: str) -> str:
    """
    Detects the category of a given piece of news using zero-shot-classification.

    Args:
        text (str): The news text to be classified.

    Returns:
        str: The category of the news ('technology', 'sports', or 'politics').
    """
    # Setup pipeline for zero-shot classification
    pipe = pipeline("zero-shot-classification")
    # Classify text and get the most probable label
    result = pipe(text, candidate_labels=['politics', 'technology', 'sports'])[0]
    return result["labels"][0].lower()


# test_function_code --------------------

def test_news_category_detection():
    assert news_category_detection('Apple just announced the newest iPhone X') == 'technology'
    assert news_category_detection('The Lakers won their last game') == 'sports'
    assert news_category_detection('The president will give a speech tomorrow') == 'politics'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_news_category_detection()