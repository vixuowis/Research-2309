# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(user_review):
    """
    Analyze the sentiment of a user review using a pre-trained model.

    Args:
        user_review (str): The user review text.

    Returns:
        dict: The sentiment analysis result, which includes the sentiment label and score.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(user_review, str):
        raise ValueError('The user review must be a string.')

    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_analyzer(user_review)
    return sentiment_result[0]

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    # Test with a positive review
    positive_review = 'Me encanta esta aplicación.'
    result = analyze_sentiment(positive_review)
    assert result['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

    # Test with a negative review
    negative_review = 'No me gusta esta aplicación.'
    result = analyze_sentiment(negative_review)
    assert result['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

    # Test with a neutral review
    neutral_review = 'Esta aplicación es normal.'
    result = analyze_sentiment(neutral_review)
    assert result['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_sentiment()