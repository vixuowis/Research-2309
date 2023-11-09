# function_import --------------------

from transformers import pipeline

# function_code --------------------

def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis on the given text using the 'cardiffnlp/twitter-xlm-roberta-base-sentiment' model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The result of the sentiment analysis. The keys are 'label' and 'score'.
    """
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    result = sentiment_task(text)
    return result

# test_function_code --------------------

def test_perform_sentiment_analysis():
    """
    Test the perform_sentiment_analysis function.
    """
    test_text = 'I really enjoyed the experience at this store.'
    result = perform_sentiment_analysis(test_text)
    assert isinstance(result, list)
    assert 'label' in result[0]
    assert 'score' in result[0]

# call_test_function_code --------------------

test_perform_sentiment_analysis()