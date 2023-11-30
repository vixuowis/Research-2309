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

    # Load pre-trained model for zero-shot classification from HuggingFace hub
    classifier = pipeline("zero-shot-classification")

    # Define labels to choose from
    labels = ["negative", "positive"]

    # Run zero-shot classification on review text and get the result
    result = classifier(review_text, labels)
    
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