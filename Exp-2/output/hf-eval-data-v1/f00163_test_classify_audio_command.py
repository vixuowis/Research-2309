def test_classify_audio_command():
    """
    This function tests the classify_audio_command function by using a sample audio file.
    """
    # Define the path to the sample audio file
    sample_audio_file_path = 'sample_audio.wav'
    
    # Call the classify_audio_command function with the sample audio file
    result = classify_audio_command(sample_audio_file_path)
    
    # Assert that the result is not None
    assert result is not None
    
    # Assert that the result is a dictionary
    assert isinstance(result, dict)
    
    # Assert that the result dictionary has a 'label' key
    assert 'label' in result
    
    # Assert that the result dictionary has a 'score' key
    assert 'score' in result

test_classify_audio_command()