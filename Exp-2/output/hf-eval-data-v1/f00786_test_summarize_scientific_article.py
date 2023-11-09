def test_summarize_scientific_article():
    article = "Here is the scientific article text..."
    summary = summarize_scientific_article(article)
    assert isinstance(summary, str), 'The result should be a string.'
    assert len(summary) < len(article), 'The summary should be shorter than the original article.'

test_summarize_scientific_article()