def test_transcribe_audio():
    '''
    This function tests the transcribe_audio function by loading a sample from the LibriSpeech dataset and comparing the output to the expected transcription.
    '''
    # Load a sample from the LibriSpeech dataset
    ds = load_dataset('librispeech_asr', 'clean', split='validation')
    sample = ds[0]['audio']
    
    # Get the expected transcription
    expected_transcription = ds[0]['text']
    
    # Transcribe the audio sample
    transcription = transcribe_audio(sample)
    
    # Compare the transcription to the expected transcription
    assert transcription.lower() in expected_transcription.lower(), f'Expected {expected_transcription}, but got {transcription}'
    
    print('Test passed.')

test_transcribe_audio()