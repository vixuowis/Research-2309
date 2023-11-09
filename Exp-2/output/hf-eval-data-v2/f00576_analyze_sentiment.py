# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review_text):
    """
    Analyze the sentiment of a given text using the 'finiteautomata/beto-sentiment-analysis' model.

    Args:
        review_text (str): The text to be analyzed.

    Returns:
        dict: The sentiment analysis result. The keys are 'label' and 'score'. 'label' can be 'POS', 'NEG', or 'NEU'. 'score' is the confidence of the prediction.
    """
    sentiment_model = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_model(review_text)
    return sentiment_result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the function analyze_sentiment.
    """
    test_text = 'Este producto es increíble.'
    result = analyze_sentiment(test_text)
    assert isinstance(result, list)
    assert 'label' in result[0]
    assert 'score' in result[0]
    assert result[0]['label'] in ['POS', 'NEG', 'NEU']

# call_test_function_code --------------------

test_analyze_sentiment()