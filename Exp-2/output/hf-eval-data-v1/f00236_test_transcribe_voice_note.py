from datasets import load_dataset

# Test function for transcribe_voice_note
# @param None
# @return: None

def test_transcribe_voice_note():
    # Load the test dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')

    # Select a sample from the dataset
    sample = ds[0]['audio']
    sampling_rate = ds[0]['sampling_rate']

    # Call the function with the sample audio and sampling rate
    transcription = transcribe_voice_note(sample, sampling_rate)

    # Assert that the transcription is not empty
    assert len(transcription) > 0, 'The transcription is empty.'

    # Assert that the transcription is a string
    assert isinstance(transcription, str), 'The transcription is not a string.'

test_transcribe_voice_note()