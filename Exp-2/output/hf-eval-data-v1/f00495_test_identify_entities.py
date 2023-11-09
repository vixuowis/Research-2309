def test_identify_entities():
    '''
    This function tests the identify_entities function.
    It uses a test dataset and asserts that the function returns the expected output.
    '''
    # Define the test dataset
    test_dataset = ['George Washington ging nach Washington', 'Angela Merkel ist die Bundeskanzlerin von Deutschland']
    
    # Define the expected output
    expected_output = [['Span [1,2]: "George Washington"   [− Labels: PER (0.9999)]', 'Span [4]: "Washington"   [− Labels: LOC (0.9997)]'], ['Span [1,2]: "Angela Merkel"   [− Labels: PER (0.9999)]', 'Span [6]: "Deutschland"   [− Labels: LOC (0.9997)]']]
    
    # Assert that the function returns the expected output
    for i, text in enumerate(test_dataset):
        assert identify_entities(text) == expected_output[i], f'For text: {text}, expected: {expected_output[i]}, but got: {identify_entities(text)}'

test_identify_entities()