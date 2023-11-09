def test_detect_voice_activity():
    """
    This function tests the detect_voice_activity function by using a sample audio file.
    """
    # Define the path to the sample audio file
    audio_file = 'sample.wav'
    
    # Call the detect_voice_activity function
    active_speech_segments = detect_voice_activity(audio_file)
    
    # Assert that the function returns a list
    assert isinstance(active_speech_segments, list), 'The function should return a list.'
    
    # Assert that the list is not empty
    assert len(active_speech_segments) > 0, 'The list should not be empty.'
    
    # Assert that each element in the list is a tuple
    for segment in active_speech_segments:
        assert isinstance(segment, tuple), 'Each element in the list should be a tuple.'
        
        # Assert that each tuple contains two elements
        assert len(segment) == 2, 'Each tuple should contain two elements.'
        
        # Assert that the first element in the tuple is less than the second element
        assert segment[0] < segment[1], 'The first element in the tuple should be less than the second element.'

# Run the test function
test_detect_voice_activity()