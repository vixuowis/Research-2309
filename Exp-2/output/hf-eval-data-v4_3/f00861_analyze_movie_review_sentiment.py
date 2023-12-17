# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_movie_review_sentiment(review: str) -> str:
    """
    Analyze the sentiment of a movie review using a pre-trained model.

    Args:
        review (str): A movie review text.

    Returns:
        str: The sentiment prediction ('POSITIVE' or 'NEGATIVE').

    Raises:
        ValueError: If the input review is empty.
    """
    if not review:
        raise ValueError("Review text cannot be empty.")
    sentiment_classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    result = sentiment_classifier(review)[0]
    return result['label']

# test_function_code --------------------

def test_analyze_movie_review_sentiment():
    print("Testing started.")
    positive_review = "This movie was fantastic! I loved it."
    negative_review = "This movie was terrible. I hated it."
    empty_review = ""

    # Test case 1: Positive review
    print("Testing case [1/3] started.")
    assert analyze_movie_review_sentiment(positive_review) == 'POSITIVE', "Test case [1/3] failed: Positive review did not return 'POSITIVE'."

    # Test case 2: Negative review
    print("Testing case [2/3] started.")
    assert analyze_movie_review_sentiment(negative_review) == 'NEGATIVE', "Test case [2/3] failed: Negative review did not return 'NEGATIVE'."

    # Test case 3: Empty review
    print("Testing case [3/3] started.")
    try:
        analyze_movie_review_sentiment(empty_review)
        assert False, "Test case [3/3] failed: No ValueError raised for empty review."
    except ValueError:
        assert True, "Test case [3/3] passed: ValueError raised for empty review."
    print("Testing finished.")

# Run the test function
test_analyze_movie_review_sentiment()

# call_test_function_line --------------------

test_analyze_movie_review_sentiment()