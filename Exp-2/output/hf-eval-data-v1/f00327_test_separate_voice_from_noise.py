def test_separate_voice_from_noise():
    """
    This function tests the separate_voice_from_noise function.
    """
    # Define the path to the test audio file
    test_audio_file_path = 'test.wav'
    
    # Call the function with the test audio file
    clean_audio_file_path = separate_voice_from_noise(test_audio_file_path)
    
    # Check that the function returns a string
    assert isinstance(clean_audio_file_path, str)
    
    # Check that the returned string is the path to a .wav file
    assert clean_audio_file_path.endswith('.wav')
    
    # Check that the returned string is different from the input string
    assert clean_audio_file_path != test_audio_file_path
    
    # Check that the returned string contains '_clean'
    assert '_clean' in clean_audio_file_path