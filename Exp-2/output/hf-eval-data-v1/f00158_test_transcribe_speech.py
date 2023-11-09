def test_transcribe_speech():
    '''
    This function tests the transcribe_speech function.
    '''
    # Load the test dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    
    # Select a sample from the dataset
    sample = ds[0]['audio']
    
    # Transcribe the sample
    transcription = transcribe_speech(sample)
    
    # Assert that the transcription is not empty
    assert len(transcription) > 0, 'The transcription is empty.'
    
    # Assert that the transcription is a string
    assert isinstance(transcription, str), 'The transcription is not a string.'

test_transcribe_speech()