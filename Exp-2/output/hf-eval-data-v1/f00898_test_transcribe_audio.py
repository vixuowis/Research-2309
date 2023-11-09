from datasets import load_dataset

def test_transcribe_audio():
    """
    Tests the transcribe_audio function.
    """
    ds = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]
    audio_data = sample['audio']['array']
    transcription = transcribe_audio(audio_data)

    assert isinstance(transcription, str), 'The transcription should be a string.'
    assert len(transcription) > 0, 'The transcription should not be empty.'

    print('All tests passed.')

test_transcribe_audio()