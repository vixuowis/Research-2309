def test_summarize_french_news():
    """
    This function tests the summarize_french_news function.
    It uses a sample French news article and checks if the output is a string.
    """
    sample_article = "L'article de presse en français ici..."
    summary = summarize_french_news(sample_article)
    assert isinstance(summary, str), "The function should return a string."

test_summarize_french_news()