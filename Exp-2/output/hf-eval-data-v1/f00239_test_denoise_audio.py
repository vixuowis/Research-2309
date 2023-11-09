def test_denoise_audio():
    '''
    This function tests the denoise_audio function.
    It uses a sample audio stream and checks if the output is a tensor.
    '''
    # Sample audio stream
    audio = torch.rand(1, 16000)
    
    # Denoise the audio
    denoised_audio = denoise_audio(audio)
    
    # Check if the output is a tensor
    assert isinstance(denoised_audio, torch.Tensor), 'The output should be a tensor.'
    
    # Check if the output has the same shape as the input
    assert denoised_audio.shape == audio.shape, 'The output and input should have the same shape.'

test_denoise_audio()