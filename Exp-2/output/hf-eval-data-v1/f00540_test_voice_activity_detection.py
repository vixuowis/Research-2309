def test_voice_activity_detection():
    # Test the voice activity detection function
    # We will use a sample audio for testing
    # The audio is obtained from an online source
    
    # Load the sample audio
    audio = torch.rand(1, 44100*10)
    
    # Perform voice activity detection on the sample audio
    voice_activity = voice_activity_detection(audio)
    
    # Check if the function returns a tensor
    assert isinstance(voice_activity, torch.Tensor), 'The function should return a tensor.'
    
    # Check if the tensor is not empty
    assert voice_activity.numel() > 0, 'The tensor should not be empty.'

test_voice_activity_detection()