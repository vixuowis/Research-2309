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
    
    # Load model
    model_name = "BSC/newscategories_bert"
    cat_model = pipeline(task="zero-shot-classification", model=model_name)

    # Classify the news text in one of 3 categories.
    results = cat_model(text, ["technology", "politics", "sports"])
    
    if 'category' not in results[0]: return ''
    
    return results[0]['category']

# test_function_code --------------------

def test_news_category_detection():
    assert news_category_detection('Apple just announced the newest iPhone X') == 'technology'
    assert news_category_detection('The Lakers won their last game') == 'sports'
    assert news_category_detection('The president will give a speech tomorrow') == 'politics'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_news_category_detection()