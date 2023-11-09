def test_translate_research_summary():
    # Test the function with a sample research summary
    research_summary = 'Climate change is a significant and lasting change in the statistical distribution of weather patterns over periods ranging from decades to millions of years.'
    translated_summary = translate_research_summary(research_summary)

    # Assert that the function returns a list
    assert isinstance(translated_summary, list)

    # Assert that the list is not empty
    assert len(translated_summary) > 0

test_translate_research_summary()