# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review: str) -> str:
    """
    Analyze the sentiment of a given text review.

    Args:
        review (str): The text review to be analyzed.

    Returns:
        str: The sentiment of the review ('POSITIVE' or 'NEGATIVE').

    Raises:
        ValueError: If the `review` is empty or None.
    """
    if not review:
        raise ValueError('The review text should not be empty.')
    sentiment_analysis = pipeline('text-classification', model='Seethal/sentiment_analysis_generic_dataset')
    result = sentiment_analysis(review)[0]
    return result['label']

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing started.")

    # Test case 1: Positive review
    print("Testing case [1/3] started.")
    positive_review = 'I love this product!'
    assert analyze_sentiment(positive_review) == 'POSITIVE', f"Test case [1/3] failed: Expected 'POSITIVE', got {analyze_sentiment(positive_review)}"

    # Test case 2: Negative review
    print("Testing case [2/3] started.")
    negative_review = 'I hate this product!'
    assert analyze_sentiment(negative_review) == 'NEGATIVE', f"Test case [2/3] failed: Expected 'NEGATIVE', got {analyze_sentiment(negative_review)}"

    # Test case 3: Empty review
    print("Testing case [3/3] started.")
    try:
        analyze_sentiment('')
        assert False, "Test case [3/3] failed: ValueError expected for empty review."
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_sentiment()