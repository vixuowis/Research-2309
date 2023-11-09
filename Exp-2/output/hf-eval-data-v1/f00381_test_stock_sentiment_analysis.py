def test_stock_sentiment_analysis():
    stock_comments = pd.Series(['Stock A is going up!', 'Looks like it\'s time to sell Stock B.', 'I wouldn\'t invest in Stock C right now.'])
    sentiment_results = stock_sentiment_analysis(stock_comments)
    assert len(sentiment_results) == len(stock_comments), 'The number of results should match the number of comments.'
    for result in sentiment_results:
        assert 'label' in result, 'Each result should have a label.'
        assert 'score' in result, 'Each result should have a score.'
    print('All tests passed.')

test_stock_sentiment_analysis()