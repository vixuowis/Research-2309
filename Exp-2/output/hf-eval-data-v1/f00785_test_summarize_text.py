def test_summarize_text():
    """
    Test the summarize_text function.
    """
    article = "In a shocking turn of events, the city's most renowned superhero was seen helping the elderly cross the street. This act of kindness has won the hearts of the city's residents, who now view the superhero in a new light."
    summary = summarize_text(article)
    assert isinstance(summary, str)
    assert len(summary) <= 130
    assert len(summary) >= 30
test_summarize_text()