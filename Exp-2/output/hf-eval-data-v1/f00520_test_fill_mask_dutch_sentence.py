def test_fill_mask_dutch_sentence():
    """
    This function tests the 'fill_mask_dutch_sentence' function.
    It uses a test sentence with a missing word and checks if the function correctly fills in the missing word.
    """
    # Test sentence with a missing word
    test_sentence = 'Hij ging naar de [MASK] om boodschappen te doen.'
    
    # Expected completed sentence
    expected_sentence = 'Hij ging naar de winkel om boodschappen te doen.'
    
    # Get the completed sentence from the function
    completed_sentence = fill_mask_dutch_sentence(test_sentence)
    
    # Check if the completed sentence is as expected
    assert completed_sentence == expected_sentence, f'Expected: {expected_sentence}, but got: {completed_sentence}'
    
    print('All tests passed.')

# Run the test function
test_fill_mask_dutch_sentence()