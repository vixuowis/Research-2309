# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using a pre-trained model.

    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: A dictionary containing the sentiment label and score.
    """
    feedback_sentiment = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = feedback_sentiment(text)
    return sentiment_result[0]

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    assert analyze_sentiment('Me encanta este producto.')['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    assert analyze_sentiment('No me gusta este servicio.')['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    assert analyze_sentiment('El producto es normal, ni bueno ni malo.')['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_sentiment()