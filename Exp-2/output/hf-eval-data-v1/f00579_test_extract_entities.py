def test_extract_entities():
    # Test the extract_entities function with a sample news article
    news_article = "Large parts of Los Angeles have been hit by power outages with electricity provider Southern California Edison pointing at high winds as the cause for the disruption. Thousands of residents..."
    entities = extract_entities(news_article)
    
    # Assert that the function returns a list
    assert isinstance(entities, list)
    
    # Assert that the list is not empty (i.e., some entities were extracted)
    assert len(entities) > 0

test_extract_entities()