def test_summarize_news_article():
    """
    Tests the summarize_news_article function.
    """
    news_article = "This is a test news article. It contains several sentences. The purpose of this article is to test the summarization function."
    summary = summarize_news_article(news_article)
    assert isinstance(summary, str), "The function should return a string."
    assert len(summary) < len(news_article), "The summary should be shorter than the original article."

test_summarize_news_article()