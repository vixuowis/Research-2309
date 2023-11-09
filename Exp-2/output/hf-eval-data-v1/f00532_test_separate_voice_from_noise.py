import torchaudio

# Test function for separate_voice_from_noise
# This function uses a sample audio file to test the separate_voice_from_noise function
# It asserts that the function returns two outputs (voice and noise)

def test_separate_voice_from_noise():
    # Load a sample audio file
    audio_file = torchaudio.datasets.LIBRISPEECH('test', download=True)[0][0]
    
    # Use the separate_voice_from_noise function
    voice, noise = separate_voice_from_noise(audio_file)
    
    # Assert that the function returns two outputs
    assert len(voice) > 0 and len(noise) > 0, 'Voice or noise not separated properly'
    
    # Assert that the outputs are not the same (i.e., the function has actually separated the voice from the noise)
    assert not torch.all(voice.eq(noise)), 'Voice and noise are not separated properly'
    
    # Assert that the outputs are of the correct type (i.e., torch.Tensor)
    assert isinstance(voice, torch.Tensor) and isinstance(noise, torch.Tensor), 'Output is not of type torch.Tensor'
    
    # Call the test function
    test_separate_voice_from_noise()