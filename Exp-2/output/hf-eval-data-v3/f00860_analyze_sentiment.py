# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review):
    """
    Analyze the sentiment of a given review using the FinBERT model.

    Args:
        review (str): The review text to analyze.

    Returns:
        dict: The sentiment analysis result.
    """
    classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    result = classifier(review)
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    review_positive = 'I love this financial service app. It has made managing my finances so much easier!'
    review_negative = 'I hate this app. It is so difficult to use and the customer service is terrible.'
    review_neutral = 'This app is okay. It does what it needs to do but nothing more.'

    result_positive = analyze_sentiment(review_positive)
    result_negative = analyze_sentiment(review_negative)
    result_neutral = analyze_sentiment(review_neutral)

    assert result_positive[0]['label'] == 'POSITIVE'
    assert result_negative[0]['label'] == 'NEGATIVE'
    assert result_neutral[0]['label'] == 'NEUTRAL'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_sentiment()