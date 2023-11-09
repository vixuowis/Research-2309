def test_generate_audio_announcement():
    '''
    This function tests the generate_audio_announcement function by providing a sample text and checking if the output .wav file is created.
    '''
    import os
    
    # Define a sample text
    sample_text = 'This is a test announcement.'
    
    # Call the function with the sample text
    generate_audio_announcement(sample_text)
    
    # Check if the output .wav file is created
    assert os.path.exists('speech.wav'), 'The output .wav file is not created.'
    
    # If the file is created, print a success message
    print('The test passed successfully.')

test_generate_audio_announcement()