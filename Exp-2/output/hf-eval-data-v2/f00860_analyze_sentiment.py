# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review):
    """
    Analyze the sentiment of a review using the FinBERT model.

    Args:
        review (str): The review to analyze.

    Returns:
        dict: The result of the sentiment analysis. The keys are 'label' and 'score'.
        'label' is the sentiment of the review ('positive', 'negative', or 'neutral').
        'score' is the confidence of the model in its prediction.
    """
    classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    result = classifier(review)
    return result[0]

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    review = 'I love this financial service app. It has made managing my finances so much easier!'
    result = analyze_sentiment(review)
    assert isinstance(result, dict)
    assert 'label' in result
    assert 'score' in result
    assert result['label'] in ['positive', 'negative', 'neutral']

# call_test_function_code --------------------

test_analyze_sentiment()