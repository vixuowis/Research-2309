def test_transcribe_audio():
    # Test the transcribe_audio function with a sample audio file
    # Note: Replace 'sample_audio.wav' with the path to a real audio file for this test to work
    transcription = transcribe_audio('sample_audio.wav')

    # Check that the transcription is a string (we can't check the exact content as it depends on the audio)
    assert isinstance(transcription, str)

    # Check that the transcription is not empty
    assert len(transcription) > 0

test_transcribe_audio()