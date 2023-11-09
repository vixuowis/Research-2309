def test_transcribe_audio():
    # Load the LibriSpeech dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    
    # Select a sample from the dataset
    sample = ds[0]['audio']
    
    # Transcribe the audio sample
    transcription = transcribe_audio(sample)
    
    # Assert that the transcription is not empty
    assert transcription != ''
    
    # Assert that the transcription is a string
    assert isinstance(transcription, str)

test_transcribe_audio()