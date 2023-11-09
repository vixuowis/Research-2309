# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(user_review):
    """
    Analyze the sentiment of a user review using the 'finiteautomata/beto-sentiment-analysis' model.

    Args:
        user_review (str): The user review text to analyze.

    Returns:
        dict: The sentiment analysis result. The result is a dictionary with 'label' and 'score' keys. The 'label' key can be 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'. The 'score' key is a float representing the confidence of the model in its prediction.
    """
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_analyzer(user_review)
    return sentiment_result[0]

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the 'analyze_sentiment' function with a sample user review.
    """
    user_review = 'Reseña del usuario aquí...'
    sentiment_result = analyze_sentiment(user_review)
    assert isinstance(sentiment_result, dict)
    assert 'label' in sentiment_result
    assert 'score' in sentiment_result
    assert sentiment_result['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

# call_test_function_code --------------------

test_analyze_sentiment()