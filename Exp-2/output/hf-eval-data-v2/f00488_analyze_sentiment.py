# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using a pre-trained model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The sentiment analysis result. The result is a dictionary with two keys: 'label' and 'score'. 'label' can be 'POSITIVE', 'NEGATIVE' or 'NEUTRAL'. 'score' is a float number indicating the confidence of the prediction.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text should be a string.')
    feedback_sentiment = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = feedback_sentiment(text)
    return sentiment_result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    test_text = 'Este producto es increíble.'
    result = analyze_sentiment(test_text)
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert 'label' in result[0]
    assert 'score' in result[0]
    assert result[0]['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

# call_test_function_code --------------------

test_analyze_sentiment()