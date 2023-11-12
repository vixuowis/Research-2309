# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_review_sentiment(review):
    """
    Classify the sentiment of a movie review.

    Args:
        review (str): The text of the movie review.

    Returns:
        str: The sentiment of the review ('POSITIVE' or 'NEGATIVE').

    Raises:
        OSError: If there is a problem loading the sentiment analysis model.
    """
    try:
        classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
        result = classifier(review)
        return result[0]['label']
    except OSError as e:
        print(f'Error loading model: {e}')
        raise

# test_function_code --------------------

def test_classify_review_sentiment():
    """
    Test the classify_review_sentiment function.
    """
    positive_review = 'I love this movie!'
    negative_review = 'I hate this movie!'
    neutral_review = 'This movie is okay.'
    assert classify_review_sentiment(positive_review) == 'POSITIVE'
    assert classify_review_sentiment(negative_review) == 'NEGATIVE'
    try:
        classify_review_sentiment(neutral_review)
    except Exception:
        pass
    else:
        raise AssertionError('Expected an exception for neutral review.')
    print('All tests passed.')

# call_test_function_code --------------------

test_classify_review_sentiment()