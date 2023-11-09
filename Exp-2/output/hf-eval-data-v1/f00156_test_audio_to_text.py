def test_audio_to_text():
    # Test the audio_to_text function with a sample audio file
    transcription = audio_to_text('sample_audio_file.wav')

    # Check that the transcription is not empty
    assert len(transcription) > 0, 'The transcription is empty.'

    # Check that the transcription is a string
    assert isinstance(transcription, str), 'The transcription is not a string.'

    # Check that the transcription is not just whitespace
    assert transcription.strip() != '', 'The transcription is just whitespace.'

test_audio_to_text()