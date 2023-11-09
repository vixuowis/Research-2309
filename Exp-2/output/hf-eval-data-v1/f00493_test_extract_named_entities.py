def test_extract_named_entities():
    '''
    This function tests the 'extract_named_entities' function by using a test dataset.
    '''
    # Define the test dataset
    test_dataset = [
        'On June 7th, Jane Smith visited the Empire State Building in New York with an entry fee of 35 dollars.',
        'On September 1st George Washington won 1 dollar.',
        'The Eiffel Tower in Paris is one of the most famous landmarks in the world.',
        'Apple Inc. is an American multinational technology company headquartered in Cupertino, California.',
        'The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.'
    ]
    
    # Test the 'extract_named_entities' function with the test dataset
    for text in test_dataset:
        entities = extract_named_entities(text)
        
        # Assert that the function returns a list
        assert isinstance(entities, list)
        
        # Assert that the function does not return an empty list
        assert len(entities) > 0

test_extract_named_entities()