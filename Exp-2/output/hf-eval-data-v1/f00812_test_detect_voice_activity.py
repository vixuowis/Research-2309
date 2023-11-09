def test_detect_voice_activity():
    """
    Tests the detect_voice_activity function by analyzing a sample audio file.
    """
    sample_audio_file_path = 'path_to_sample_audio_file'
    voice_activity = detect_voice_activity(sample_audio_file_path)
    assert isinstance(voice_activity, dict), 'The result should be a dictionary.'
    assert 'speech' in voice_activity, 'The result should contain a speech key.'
    assert 'non_speech' in voice_activity, 'The result should contain a non_speech key.'

test_detect_voice_activity()