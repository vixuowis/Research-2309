# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_review_sentiment(review_text):
    """
    Analyze the sentiment of a product review using a pre-trained model.

    Args:
        review_text (str): The text of the review to analyze.

    Returns:
        dict: The sentiment analysis result, including the label ('1 star' to '5 stars') and the score.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    review_sentiment = sentiment_pipeline(review_text)
    return review_sentiment[0]

# test_function_code --------------------

def test_analyze_review_sentiment():
    """
    Test the analyze_review_sentiment function with some example reviews.
    """
    positive_review = 'I love this product!'
    negative_review = 'I hate this product!'
    neutral_review = 'This product is okay.'

    positive_result = analyze_review_sentiment(positive_review)
    negative_result = analyze_review_sentiment(negative_review)
    neutral_result = analyze_review_sentiment(neutral_review)

    assert 'stars' in positive_result['label'] and positive_result['score'] > 0.5
    assert 'stars' in negative_result['label'] and negative_result['score'] > 0.5
    assert 'stars' in neutral_result['label'] and neutral_result['score'] > 0.5

# call_test_function_code --------------------

test_analyze_review_sentiment()