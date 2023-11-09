def test_voice_activity_detection():
    """
    This function tests the 'voice_activity_detection' function with a sample audio file.
    """

    # Specify the path to the sample audio file
    audio_file = 'TheBigBangTheory.wav'

    # Call the 'voice_activity_detection' function with the sample audio file
    result = voice_activity_detection(audio_file)

    # Assert that the result is not None
    assert result is not None

    # Assert that the result is a dictionary
    assert isinstance(result, dict)

    # Assert that the dictionary contains the 'audio' key
    assert 'audio' in result

test_voice_activity_detection()