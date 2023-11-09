# Test function for analyze_review_sentiment
# This function tests the analyze_review_sentiment function with a sample review.
# It uses the assert statement to verify that the function returns a result (not None).
def test_analyze_review_sentiment():
    sample_review = 'I love this product!'
    sentiment_score = analyze_review_sentiment(sample_review)
    assert sentiment_score is not None, 'The function should return a sentiment score.'
    print('Test passed.')

test_analyze_review_sentiment()