def test_extract_audio_features():
    """
    This function tests the 'extract_audio_features' function by using a sample audio file.
    """
    # Define the path to the sample audio file
    sample_audio = 'sample_audio.wav'
    
    # Extract features from the sample audio file
    features = extract_audio_features(sample_audio)
    
    # Assert that the features are not None
    assert features is not None, 'The extracted features should not be None.'
    
    # Assert that the features are of the correct type (Tensor)
    assert isinstance(features, torch.Tensor), 'The extracted features should be a Tensor.'

test_extract_audio_features()