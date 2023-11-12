# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review_text):
    """
    Analyze the sentiment of a given text using the 'finiteautomata/beto-sentiment-analysis' model.

    Args:
        review_text (str): The text to be analyzed.

    Returns:
        dict: The sentiment analysis result. The keys are 'label' and 'score'.
    """
    sentiment_model = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_model(review_text)
    return sentiment_result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the function analyze_sentiment.
    """
    assert analyze_sentiment('Me encanta este producto.')[0]['label'] == 'POS'
    assert analyze_sentiment('No me gusta este producto.')[0]['label'] == 'NEG'
    assert analyze_sentiment('Este producto es normal.')[0]['label'] == 'NEU'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_sentiment()