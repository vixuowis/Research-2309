def test_separate_speaker_sources():
    """
    This function tests the separate_speaker_sources function by loading a sample audio file and checking the output.
    """
    sep_sources = separate_speaker_sources('sample_audio.wav')
    assert isinstance(sep_sources, np.ndarray), 'The output should be a numpy array.'
    assert sep_sources.shape[0] > 0, 'The output array should not be empty.'

test_separate_speaker_sources()