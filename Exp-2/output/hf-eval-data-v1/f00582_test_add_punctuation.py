def test_add_punctuation():
    """
    This function tests the add_punctuation function.
    It uses a sample message and checks if the output is a string.
    """
    # Define a sample message
    sample_message = 'hello how are you'
    
    # Apply the add_punctuation function to the sample message
    corrected_message = add_punctuation(sample_message)
    
    # Check if the output is a string
    assert isinstance(corrected_message, str), 'The output should be a string.'
    
    # Check if the output is not the same as the input (i.e., punctuation has been added)
    assert corrected_message != sample_message, 'The output should not be the same as the input.'

test_add_punctuation()