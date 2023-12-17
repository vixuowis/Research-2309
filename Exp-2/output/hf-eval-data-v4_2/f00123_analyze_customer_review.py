# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_review(review_text):
    """
    Analyzes the sentiment of a given customer review text.

    Args:
        review_text (str): The customer review text to analyze.

    Returns:
        dict: A dictionary containing the result of the sentiment analysis.

    Raises:
        ValueError: If the input review_text is not a string.
    """
    if not isinstance(review_text, str):
        raise ValueError('Input must be a string.')
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(review_text)
    return result

# test_function_code --------------------

def test_analyze_customer_review():
    print("Testing started.")

    # Test case 1: Check if function returns correct type
    print("Testing case [1/3] started.")
    result_type = isinstance(analyze_customer_review("Great product!"), list)
    assert result_type, f"Test case [1/3] failed: Function should return a list, but got {type(result_type)}"

    # Test case 2: Check if ValueError is raised for non-string input
    print("Testing case [2/3] started.")
    try:
        analyze_customer_review(None)
        assert False, "Test case [2/3] failed: Function should raise ValueError for non-string input"
    except ValueError as e:
        assert str(e) == 'Input must be a string.', f"Test case [2/3] failed: {str(e)}"

    # Test case 3: Check if function returns a dictionary with expected keys
    print("Testing case [3/3] started.")
    result_keys = analyze_customer_review("Not recommended").pop().keys()
    expected_keys = {'label', 'score'}
    assert result_keys == expected_keys, f"Test case [3/3] failed: Function should return dictionary with keys {expected_keys}, but got {result_keys}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_review()