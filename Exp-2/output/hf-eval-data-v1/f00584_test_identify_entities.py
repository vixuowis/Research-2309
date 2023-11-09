def test_identify_entities():
    '''
    This function tests the 'identify_entities' function.
    It uses a test dataset and asserts that the function returns the expected output.
    '''
    # Define a test dataset
    test_dataset = [
        ('George Washington went to Washington', ['George Washington', 'Washington']),
        ('I visited Paris and met John Doe', ['Paris', 'John Doe']),
        ('Barack Obama was in the White House', ['Barack Obama', 'White House'])
    ]
    
    # Iterate over the test dataset
    for diary_entry_text, expected_output in test_dataset:
        # Call the 'identify_entities' function and store the output
        output = identify_entities(diary_entry_text)
        
        # Assert that the output is as expected
        assert set(output) == set(expected_output), f'For input {diary_entry_text}, expected {expected_output} but got {output}'
    
    print('All test cases pass')
    
# Call the test function
test_identify_entities()