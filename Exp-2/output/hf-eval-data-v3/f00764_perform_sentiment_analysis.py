# function_import --------------------

from transformers import pipeline

# function_code --------------------

def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis on the input text using a pre-trained model.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: The sentiment analysis result.
    """
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    result = sentiment_task(text)
    return result

# test_function_code --------------------

def test_perform_sentiment_analysis():
    """
    Test the perform_sentiment_analysis function.
    """
    assert isinstance(perform_sentiment_analysis('I really enjoyed the experience at this store.'), list)
    assert isinstance(perform_sentiment_analysis('I am not happy with the service.'), list)
    assert isinstance(perform_sentiment_analysis('The product is just okay.'), list)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_perform_sentiment_analysis()