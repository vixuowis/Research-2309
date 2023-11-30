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
    
    # define pipeline for zero shot classification
    zero_shot_classification = pipeline("zero-shot-classification")

    # classify text with labels
    result = zero_shot_classification(text, candidate_labels=["technology", "sports", "politics"])
    
    # extract category label
    return result['labels'][0]

# test_function_code --------------------

def test_news_category_detection():
    assert news_category_detection('Apple just announced the newest iPhone X') == 'technology'
    assert news_category_detection('The Lakers won their last game') == 'sports'
    assert news_category_detection('The president will give a speech tomorrow') == 'politics'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_news_category_detection()