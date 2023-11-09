def test_measure_noise_levels():
    """
    Tests the measure_noise_levels function.
    """
    # Use a sample audio file for testing
    audio_file = 'sample.wav'
    # Use a dummy access token for testing
    access_token = 'dummy_token'
    results = measure_noise_levels(audio_file, access_token)
    # Check that the results is a dictionary
    assert isinstance(results, dict)
    # Check that the dictionary is not empty
    assert results
    # Check that each frame in the results has the expected keys
    for frame in results:
        assert 'vad' in results[frame]
        assert 'snr' in results[frame]
        assert 'c50' in results[frame]