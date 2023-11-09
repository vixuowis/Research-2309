def test_extract_entities():
    """
    Test the function extract_entities.
    """
    test_article = "On September 1st George Washington won 1 dollar."
    entities = extract_entities(test_article)
    assert len(entities) > 0
    assert 'George Washington' in [entity.text for entity in entities]

test_extract_entities()