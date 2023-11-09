def test_classify_news_article():
    """
    Test the classify_news_article function with some example news articles.
    """
    news_article1 = 'The government passed a new law today'
    assert classify_news_article(news_article1) == 'Politics'

    news_article2 = 'The local team won the football match'
    assert classify_news_article(news_article2) == 'Sports'

    news_article3 = 'Apple released a new iPhone today'
    assert classify_news_article(news_article3) == 'Technology'

    news_article4 = 'The stock market crashed today'
    assert classify_news_article(news_article4) == 'Business'

    news_article5 = 'The new movie is a blockbuster hit'
    assert classify_news_article(news_article5) == 'Entertainment'

test_classify_news_article()