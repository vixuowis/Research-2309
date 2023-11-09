def test_detect_entities():
    '''
    This function tests the detect_entities function.
    
    It uses a test sentence and checks if the detected entities match the expected entities.
    '''
    # Define the test sentence and the expected entities
    test_sentence = 'On September 1st George won 1 dollar while watching Game of Thrones.'
    expected_entities = ['September 1st', 'George', '1 dollar', 'Game of Thrones']
    
    # Call the detect_entities function with the test sentence
    detected_entities = detect_entities(test_sentence)
    
    # Check if the detected entities match the expected entities
    assert len(detected_entities) == len(expected_entities), 'Number of detected entities does not match number of expected entities.'
    for i in range(len(detected_entities)):
        assert str(detected_entities[i]) in expected_entities, f'Entity {detected_entities[i]} was not expected.'
    
    print('All tests passed.')

test_detect_entities()