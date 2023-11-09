def test_transcribe_audio():
    # Load the test dataset
    ds = load_dataset('librispeech_asr', 'clean', split='validation')

    # Select a sample from the dataset
    sample = ds[0]['audio']

    # Transcribe the audio sample
    transcription = transcribe_audio(sample)

    # Assert that the transcription is not empty
    assert len(transcription) > 0

    # Assert that the transcription is a string
    assert isinstance(transcription, str)

test_transcribe_audio()