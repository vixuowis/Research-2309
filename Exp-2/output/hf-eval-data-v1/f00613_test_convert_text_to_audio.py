def test_convert_text_to_audio():
    """
    This function tests the convert_text_to_audio function by providing a sample text and checking if the output audio file is created.
    """
    # Define a sample book text
    sample_text = 'This is a sample book text.'
    
    # Call the convert_text_to_audio function
    convert_text_to_audio(sample_text)
    
    # Check if the output audio file is created
    assert os.path.exists('audiobook_output.wav'), 'The output audio file is not created.'
    
    # If the output audio file is created, print a success message
    print('The convert_text_to_audio function passed the test.')

test_convert_text_to_audio()