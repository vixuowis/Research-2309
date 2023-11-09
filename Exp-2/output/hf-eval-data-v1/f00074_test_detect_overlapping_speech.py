def test_detect_overlapping_speech():
    """
    This function tests the detect_overlapping_speech function.
    It uses a sample audio file and checks if the function returns the expected output.
    """
    # Use a sample audio file for testing
    audio_file = 'sample_audio.wav'
    access_token = 'ACCESS_TOKEN_GOES_HERE'
    # Call the function with the sample audio file
    result = detect_overlapping_speech(audio_file, access_token)
    # Check if the function returns a list
    assert isinstance(result, list), 'The function should return a list.'
    # Check if the list contains tuples
    for segment in result:
        assert isinstance(segment, tuple), 'Each element in the list should be a tuple.'
        assert len(segment) == 2, 'Each tuple should contain two elements.'
        assert isinstance(segment[0], float) and isinstance(segment[1], float), 'Each element in the tuple should be a float.'
    print('All tests passed.')

test_detect_overlapping_speech()