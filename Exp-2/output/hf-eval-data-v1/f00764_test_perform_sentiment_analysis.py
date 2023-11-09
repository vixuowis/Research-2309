def test_perform_sentiment_analysis():
    """
    Test the perform_sentiment_analysis function.
    """
    text = 'I really enjoyed the experience at this store.'
    result = perform_sentiment_analysis(text)
    assert isinstance(result, list)
    assert 'label' in result[0]
    assert 'score' in result[0]

test_perform_sentiment_analysis()