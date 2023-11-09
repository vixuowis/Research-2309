def test_convert_text_to_speech():
    '''
    This function tests the 'convert_text_to_speech' function by providing a sample Chinese text and checking if the output audio file is created.
    '''
    # Sample Chinese text
    sample_text = '汉语很有趣'
    
    # Call the function with the sample text
    convert_text_to_speech(sample_text)
    
    # Check if the output audio file is created
    assert os.path.exists('lesson_audio_example.wav')

test_convert_text_to_speech()