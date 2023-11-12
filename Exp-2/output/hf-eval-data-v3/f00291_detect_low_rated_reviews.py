# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_low_rated_reviews(review_text):
    """
    Detects low-rated product reviews using a sentiment analysis model.

    Args:
        review_text (str): The text of the product review.

    Returns:
        bool: True if the review is low-rated (less than 3 stars), False otherwise.

    Raises:
        ValueError: If the sentiment analysis model returns an unexpected result.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(review_text)
    try:
        return int(result[0]['label'][-1]) < 3
    except ValueError:
        raise ValueError('Unexpected result from sentiment analysis model')

# test_function_code --------------------

def test_detect_low_rated_reviews():
    """Tests the detect_low_rated_reviews function."""
    assert detect_low_rated_reviews('I love this product!') == False
    assert detect_low_rated_reviews('This product is terrible!') == True
    assert detect_low_rated_reviews('This product is okay.') == False
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_low_rated_reviews()