def test_analyze_review_sentiment():
    """
    Test the analyze_review_sentiment function.
    """
    # Define a positive and a negative review
    positive_review = 'The food was delicious and the service was excellent.'
    negative_review = 'The food was terrible and the service was poor.'

    # Test the function
    assert analyze_review_sentiment(positive_review) == 'Positive'
    assert analyze_review_sentiment(negative_review) == 'Negative'