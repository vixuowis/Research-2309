# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_movie_review_sentiment(review_text):
    """
    Analyze the sentiment of the movie review using zero-shot classification.

    Args:
        review_text (str): A text string containing the movie review to be analyzed.

    Returns:
        dict: A dictionary containing the classification results with probabilities.

    Raises:
        ValueError: If the review_text is empty or not a string.
    """
    if not review_text or not isinstance(review_text, str):
        raise ValueError('The review text must be a non-empty string.')

    # Initialize the zero-shot classification pipeline
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    # Define the candidate labels
    candidate_labels = ['positive', 'negative']
    # Classify the review text
    classification_result = nlp(review_text, candidate_labels)
    return classification_result

# test_function_code --------------------

def test_analyze_movie_review_sentiment():
    print("Testing started.")

    # Test case 1: Positive review
    print("Testing case [1/3] started.")
    positive_review = "The movie 'Inception' is an exceptional piece of cinematic art."
    positive_result = analyze_movie_review_sentiment(positive_review)
    assert positive_result['labels'][0] == 'positive', f"Test case [1/3] failed: {positive_result}"

    # Test case 2: Negative review
    print("Testing case [2/3] started.")
    negative_review = "The movie 'Inception' was a total disappointment."
    negative_result = analyze_movie_review_sentiment(negative_review)
    assert negative_result['labels'][0] == 'negative', f"Test case [2/3] failed: {negative_result}"

    # Test case 3: Empty review
    print("Testing case [3/3] started.")
    empty_review = ""
    try:
        analyze_movie_review_sentiment(empty_review)
        assert False, "Test case [3/3] failed: Exception should have been raised for empty review."
    except ValueError as e:
        assert str(e) == 'The review text must be a non-empty string.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_movie_review_sentiment()