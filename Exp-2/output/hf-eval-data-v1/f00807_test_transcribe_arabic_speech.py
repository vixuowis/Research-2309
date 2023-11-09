def test_transcribe_arabic_speech():
    """
    Tests the transcribe_arabic_speech function by transcribing a small sample of Arabic speech.
    """
    # Replace with the path to your test audio file
    test_audio_path = '/path/to/test_audio.wav'
    transcription = transcribe_arabic_speech([test_audio_path])
    print(transcription)
    assert isinstance(transcription, list), 'The output should be a list.'
    assert len(transcription) > 0, 'The output list should not be empty.'

test_transcribe_arabic_speech()