def test_summarize_article():
    """
    This function tests the 'summarize_article' function by using a sample article.
    """
    # Define a sample article
    article = "Long article text here..."
    
    # Get the summary of the article
    summary = summarize_article(article)
    
    # Check if the summary is not empty
    assert summary != '', 'The summary is empty.'
    
    # Check if the summary is indeed shorter than the original article
    assert len(summary) < len(article), 'The summary is not shorter than the original article.'

test_summarize_article()