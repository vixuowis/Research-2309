# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review):
    """
    Analyze the sentiment of a given review using a multilingual sentiment analysis model.

    Args:
        review (str): The review to be analyzed. It can be in any of the six languages: English, Dutch, German, French, Spanish and Italian.

    Returns:
        dict: A dictionary containing the sentiment analysis result. It includes the label (positive or negative) and the score.

    Raises:
        ValueError: If the review is not a string.
    """
    if not isinstance(review, str):
        raise ValueError('Review must be a string.')

    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(review)
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function with some sample reviews.
    """
    review_english = 'I love this product!'
    review_spanish = '¡Esto es maravilloso! Me encanta.'
    result_english = analyze_sentiment(review_english)
    result_spanish = analyze_sentiment(review_spanish)

    assert isinstance(result_english, list) and isinstance(result_english[0], dict)
    assert isinstance(result_spanish, list) and isinstance(result_spanish[0], dict)
    assert 'label' in result_english[0] and 'score' in result_english[0]
    assert 'label' in result_spanish[0] and 'score' in result_spanish[0]

# call_test_function_code --------------------

test_analyze_sentiment()