def test_find_best_bard():
    '''
    This function tests the 'find_best_bard' function by using a sample dataset.
    '''
    # A sample dataset.
    table_data = {
        'Bard': ['Bard1', 'Bard2', 'Bard3'],
        'Magical Ability': [50, 75, 100]
    }
    
    # The expected result.
    expected_result = 'Bard3'
    
    # Call the 'find_best_bard' function with the sample dataset.
    result = find_best_bard(table_data)
    
    # Assert that the result is as expected.
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

test_find_best_bard()