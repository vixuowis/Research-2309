def test_summarize_news_article():
    '''
    This function tests the 'summarize_news_article' function by using a sample news article text.
    It asserts that the output is a string and that it is not the same as the input text, indicating that summarization has occurred.
    '''
    article_text = 'International news article text here...'
    summary = summarize_news_article(article_text)
    assert isinstance(summary, str)
    assert summary != article_text
test_summarize_news_article()