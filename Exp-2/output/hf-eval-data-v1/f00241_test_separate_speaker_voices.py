def test_separate_speaker_voices():
    """
    This function tests the 'separate_speaker_voices' function.
    """
    # Load a sample mixed audio recording
    wavs = np.random.rand(8000)
    
    # Use the function to separate the speaker voices
    separated_audio = separate_speaker_voices(wavs)
    
    # Check the output type
    assert isinstance(separated_audio, np.ndarray), "Output should be a numpy array."
    
    # Check the output shape
    assert separated_audio.shape[0] == 3, "Output should have 3 channels."

test_separate_speaker_voices()