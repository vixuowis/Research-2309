def test_extract_entities():
    # Test the function with some text
    text = "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne to develop and sell Wozniak's Apple I personal computer."
    entities = extract_entities(text)
    
    # Assert that the function returns the correct entities
    assert 'Apple' in entities
    assert 'Steve Jobs' in entities
    assert 'Steve Wozniak' in entities
    assert 'Ronald Wayne' in entities
    
    # Test the function with some other text
    text = "Microsoft was founded by Bill Gates and Paul Allen."
    entities = extract_entities(text)
    
    # Assert that the function returns the correct entities
    assert 'Microsoft' in entities
    assert 'Bill Gates' in entities
    assert 'Paul Allen' in entities

test_extract_entities()