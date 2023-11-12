# function_import --------------------

from transformers import pipeline

# function_code --------------------

def sentiment_analysis(text):
    """
    Perform sentiment analysis on the given text using the 'nlptown/bert-base-multilingual-uncased-sentiment' model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The sentiment analysis result.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_result = sentiment_pipeline(text)
    return sentiment_result

# test_function_code --------------------

def test_sentiment_analysis():
    """
    Test the sentiment_analysis function.
    """
    positive_text = 'The book is well-written, engaging, and insightful.'
    negative_text = 'The book is boring and uninteresting.'
    neutral_text = 'The book is okay.'

    positive_result = sentiment_analysis(positive_text)
    negative_result = sentiment_analysis(negative_text)
    neutral_result = sentiment_analysis(neutral_text)

    assert positive_result[0]['label'] in ['4 stars', '5 stars'], 'Test Failed: Expected 4 or 5 stars for positive text.'
    assert negative_result[0]['label'] in ['1 star', '2 stars'], 'Test Failed: Expected 1 or 2 stars for negative text.'
    assert neutral_result[0]['label'] in ['3 stars'], 'Test Failed: Expected 3 stars for neutral text.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_sentiment_analysis()