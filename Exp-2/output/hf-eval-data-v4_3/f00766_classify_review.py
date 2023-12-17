# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_review(review_text):
    """
    Classify the sentiment of a movie review using a pre-trained model.

    Args:
        review_text (str): The text of the movie review to be classified.

    Returns:
        dict: A dictionary containing the label ('LABEL_0' for negative, 'LABEL_1' for positive) and the associated score.

    Raises:
        ValueError: If the review_text is not a string or is empty.
    """
    if not isinstance(review_text, str) or not review_text:
        raise ValueError("The review text must be a non-empty string.")
    classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    return classifier(review_text)[0]

# test_function_code --------------------

def test_classify_review():
    print("Testing started.")

    # Test case 1: Positive review
    print("Testing case [1/3] started.")
    positive_review = "I absolutely loved this movie!"
    result_positive = classify_review(positive_review)
    assert result_positive['label'] == 'LABEL_1', f"Test case [1/3] failed: Expected 'LABEL_1', got {result_positive['label']}"

    # Test case 2: Negative review
    print("Testing case [2/3] started.")
    negative_review = "This movie was terrible and boring."
    result_negative = classify_review(negative_review)
    assert result_negative['label'] == 'LABEL_0', f"Test case [2/3] failed: Expected 'LABEL_0', got {result_negative['label']}"

    # Test case 3: Invalid review
    print("Testing case [3/3] started.")
    invalid_review = ""
    try:
        classify_review(invalid_review)
        assert False, "Test case [3/3] failed: ValueError exception expected"
    except ValueError as e:
        assert str(e) == "The review text must be a non-empty string.", f"Test case [3/3] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_review()