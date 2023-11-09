from datasets import load_dataset

# Test function for convert_audio_to_text
def test_convert_audio_to_text():
    '''
    This function tests the convert_audio_to_text function by loading a sample from the LibriSpeech dataset
    and comparing the output of the function to the expected transcription.
    '''
    # Load a sample from the LibriSpeech dataset
    ds = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    audio_file = ds[0]['audio']['array']
    expected_transcription = ds[0]['text']

    # Get the transcription from the function
    transcription = convert_audio_to_text(audio_file)

    # Compare the transcription to the expected transcription
    assert transcription == expected_transcription, f'Expected: {expected_transcription}, but got: {transcription}'

# Run the test function
test_convert_audio_to_text()