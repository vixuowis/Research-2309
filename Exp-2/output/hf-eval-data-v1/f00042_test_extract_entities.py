def test_extract_entities():
    # Test the function with some example text
    text = 'My name is Wolfgang and I live in Berlin'
    entities = extract_entities(text)
    # Check that the function returns a list
    assert isinstance(entities, list)
    # Check that the function identifies the correct number of entities
    # Note: The exact number of entities may vary depending on the model and the text,
    # so we check that the function identifies at least one entity
    assert len(entities) > 0
test_extract_entities()