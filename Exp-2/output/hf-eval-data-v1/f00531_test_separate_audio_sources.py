def test_separate_audio_sources():
    '''
    This function tests the separate_audio_sources function by separating the sources in a sample audio file and checking if the output files are created.
    '''
    # Separate the sources in the sample audio file
    separate_audio_sources('speechbrain/sepformer-wsj02mix/test_mixture.wav')
    
    # Check if the output files are created
    assert os.path.exists('source1hat.wav')
    assert os.path.exists('source2hat.wav')

test_separate_audio_sources()