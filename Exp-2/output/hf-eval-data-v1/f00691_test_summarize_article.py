def test_summarize_article():
    """
    This function tests the summarize_article function by using a sample article.
    """
    # Define a sample article
    article = 'Studies have shown that owning a dog is good for you. Dogs are known to reduce stress, increase fitness levels, and improve overall happiness.'
    
    # Call the summarize_article function
    summary = summarize_article(article)
    
    # Assert that the summary is not None
    assert summary is not None, 'The summary should not be None.'
    
    # Assert that the summary is shorter than the original article
    assert len(summary) < len(article), 'The summary should be shorter than the original article.'

test_summarize_article()