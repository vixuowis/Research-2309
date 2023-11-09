# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(message):
    """
    Analyze the sentiment of a given message using the 'cardiffnlp/twitter-xlm-roberta-base-sentiment' model.

    Args:
        message (str): The message to analyze.

    Returns:
        dict: The result of the sentiment analysis. The keys are 'label' and 'score'.
    """
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    sentiment_analysis_result = sentiment_task(message)
    return sentiment_analysis_result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    message = 'I am really frustrated with the service'
    result = analyze_sentiment(message)
    assert isinstance(result, list)
    assert 'label' in result[0]
    assert 'score' in result[0]
    assert 0 <= result[0]['score'] <= 1

# call_test_function_code --------------------

test_analyze_sentiment()