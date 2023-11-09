def test_generate_summary():
    """
    Test the generate_summary function.
    """
    article_text = 'In a major breakthrough, scientists at the University of California have developed a new form of renewable energy.'
    summary = generate_summary(article_text)
    assert isinstance(summary, str), 'The result should be a string.'
    assert len(summary) <= 50, 'The length of the summary should not exceed the maximum length.'

test_generate_summary()