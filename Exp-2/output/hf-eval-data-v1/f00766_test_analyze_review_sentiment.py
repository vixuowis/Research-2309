def test_analyze_review_sentiment():
    """
    Test the analyze_review_sentiment function.
    """
    positive_review = 'I love this movie!'
    negative_review = 'I hate this movie!'

    assert analyze_review_sentiment(positive_review) == 'POSITIVE'
    assert analyze_review_sentiment(negative_review) == 'NEGATIVE'

test_analyze_review_sentiment()