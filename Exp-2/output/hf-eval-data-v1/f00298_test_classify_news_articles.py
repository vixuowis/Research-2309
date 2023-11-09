def test_classify_news_articles():
    '''
    This function tests the classify_news_articles function by using a sample news article.
    '''
    news_article = 'Apple just announced the newest iPhone X'
    assert classify_news_articles(news_article) in ['technology', 'sports', 'politics']

test_classify_news_articles()