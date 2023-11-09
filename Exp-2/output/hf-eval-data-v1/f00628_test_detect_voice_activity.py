def test_detect_voice_activity():
    """
    This function tests the 'detect_voice_activity' function by using a sample audio file.
    """
    # Path to the sample audio file
    audio_file = 'TheBigBangTheory.wav'

    # Call the 'detect_voice_activity' function with the sample audio file
    result = detect_voice_activity(audio_file)

    # Assert that the result is not None
    assert result is not None

    # Assert that the result is a dictionary
    assert isinstance(result, dict)

    # Assert that the 'audio' key is in the result dictionary
    assert 'audio' in result

test_detect_voice_activity()