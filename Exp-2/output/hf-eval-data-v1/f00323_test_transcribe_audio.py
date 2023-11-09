def test_transcribe_audio():
    # Load the test dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')

    # Select a sample from the dataset
    sample = ds[0]['audio']

    # Transcribe the audio sample
    transcription = transcribe_audio(sample['array'], sample['sampling_rate'])

    # Assert that the transcription is not empty
    assert len(transcription) > 0

    # Assert that the transcription is a list
    assert isinstance(transcription, list)

    # Assert that the transcription contains strings
    assert all(isinstance(item, str) for item in transcription)

test_transcribe_audio()