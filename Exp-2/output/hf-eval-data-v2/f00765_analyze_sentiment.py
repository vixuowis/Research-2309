# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    Perform sentiment analysis on the provided text using the 'nlptown/bert-base-multilingual-uncased-sentiment' model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The result of the sentiment analysis. The result is a dictionary with the keys 'label' and 'score'. 'label' is the predicted sentiment (1-5 stars), and 'score' is the confidence of the prediction.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_result = sentiment_pipeline(text)
    return sentiment_result[0]

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    test_text = 'The book is well-written, engaging, and insightful, but some parts feel rushed.'
    result = analyze_sentiment(test_text)
    assert isinstance(result, dict)
    assert 'label' in result
    assert 'score' in result
    assert 1 <= int(result['label'].split(' ')[0]) <= 5

# call_test_function_code --------------------

test_analyze_sentiment()