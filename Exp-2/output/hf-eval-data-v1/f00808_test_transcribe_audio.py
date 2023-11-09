def test_transcribe_audio():
    """
    Tests the transcribe_audio function.
    """
    # Load a sample audio file from the LibriSpeech (clean) test set
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']

    # Transcribe the audio file
    transcription = transcribe_audio(sample)

    # Check that the transcription is a string
    assert isinstance(transcription, str)

    # Check that the transcription is not empty
    assert len(transcription) > 0