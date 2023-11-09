def test_detect_overlapping_speech():
    """
    This function tests the detect_overlapping_speech function.
    It uses a sample audio file and checks if the function returns the expected output.
    """
    # Define the path to the sample audio file and the access token
    audio_file = 'sample_audio.wav'
    access_token = 'SAMPLE_ACCESS_TOKEN'

    # Call the function with the sample inputs
    result = detect_overlapping_speech(audio_file, access_token)

    # Since the exact output is unknown, we can only check if the function returns a list
    assert isinstance(result, list), 'The function should return a list.'

    # If the list is not empty, check if it contains tuples
    if result:
        assert isinstance(result[0], tuple), 'The list should contain tuples.'
        assert len(result[0]) == 2, 'Each tuple should contain two elements.'

    print('All tests passed.')

test_detect_overlapping_speech()