# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    This function uses a pre-trained model from the transformers library to analyze the sentiment of a given text.
    The model used is 'siebert/sentiment-roberta-large-english', which is a fine-tuned checkpoint of RoBERTa-large.
    It predicts either positive (1) or negative (0) sentiment for the given text.

    Args:
        text (str): The text to analyze sentiment for.

    Returns:
        dict: A dictionary containing the label ('POSITIVE' or 'NEGATIVE') and the score.
    """
    sentiment_analysis = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
    result = sentiment_analysis(text)
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    This function tests the analyze_sentiment function by analyzing the sentiment of a sample text.
    The expected result is 'POSITIVE' as the sample text is positive.
    """
    sample_text = 'I love the new product!'
    result = analyze_sentiment(sample_text)
    assert result[0]['label'] == 'POSITIVE', 'The sentiment analysis is incorrect.'

# call_test_function_code --------------------

test_analyze_sentiment()