def test_analyze_sentiment():
    review = 'I love this financial service app. It has made managing my finances so much easier!'
    sentiment = analyze_sentiment(review)
    assert sentiment in ['positive', 'negative', 'neutral'], 'Invalid sentiment'