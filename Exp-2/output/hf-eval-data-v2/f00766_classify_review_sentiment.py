# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_review_sentiment(review_text):
    """
    Classify the sentiment of a movie review using the 'lvwerra/distilbert-imdb' model from Hugging Face Transformers.

    Args:
        review_text (str): The text of the movie review to be classified.

    Returns:
        str: The sentiment of the review ('positive' or 'negative').

    Raises:
        ValueError: If the review_text is not a string.
    """
    if not isinstance(review_text, str):
        raise ValueError('Review text must be a string.')

    classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    result = classifier(review_text)
    return result[0]['label']

# test_function_code --------------------

def test_classify_review_sentiment():
    """
    Test the classify_review_sentiment function with some example movie reviews.
    """
    positive_review = 'I love this movie!'
    negative_review = 'I hate this movie!'

    assert classify_review_sentiment(positive_review) == 'POSITIVE'
    assert classify_review_sentiment(negative_review) == 'NEGATIVE'

# call_test_function_code --------------------

test_classify_review_sentiment()