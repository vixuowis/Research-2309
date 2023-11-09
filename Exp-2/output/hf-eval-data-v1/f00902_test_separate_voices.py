def test_separate_voices():
    """
    This function tests the separate_voices function.
    """
    # Load a sample single-channel audio recording
    audio = np.random.rand(8000)
    separated_sources = separate_voices(audio)
    assert separated_sources.shape == (2, 8000), 'The output shape is not correct.'