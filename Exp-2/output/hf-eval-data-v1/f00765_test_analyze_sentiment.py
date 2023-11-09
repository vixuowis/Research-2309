def test_analyze_sentiment():
    """
    Test the analyze_sentiment function.
    """
    test_text = 'The book is well-written, engaging, and insightful, but some parts feel rushed.'
    result = analyze_sentiment(test_text)
    assert isinstance(result, dict)
    assert 'label' in result
    assert 'score' in result
    assert 1 <= int(result['label'].split(' ')[0]) <= 5

test_analyze_sentiment()