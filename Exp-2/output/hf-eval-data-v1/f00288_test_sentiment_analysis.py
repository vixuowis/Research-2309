# This is a test function for the 'sentiment_analysis' function.
# It uses a sample review to test the function.
# The function asserts that the sentiment of the review is not None, as the sentiment analysis function should always return a sentiment.
def test_sentiment_analysis():
    review = 'I love this product!'
    assert sentiment_analysis(review) is not None

test_sentiment_analysis()