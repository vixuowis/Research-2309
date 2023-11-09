def test_separate_speakers():
    """
    This function tests the separate_speakers function.
    """
    # Test audio file
    test_audio_file = 'test_audio.wav'
    
    # Separate speakers
    output_files = separate_speakers(test_audio_file)
    
    # Check that the output is a list
    assert isinstance(output_files, list), 'Output should be a list.'
    
    # Check that the list contains strings
    for output_file in output_files:
        assert isinstance(output_file, str), 'Each item in the output list should be a string.'

test_separate_speakers()