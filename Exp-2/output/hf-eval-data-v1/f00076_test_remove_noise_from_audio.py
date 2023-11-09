def test_remove_noise_from_audio():
    """
    This function tests the remove_noise_from_audio function by using a sample noisy audio file.
    """
    # Define the path to the sample noisy audio file
    sample_noisy_audio_path = 'speechbrain/metricgan-plus-voicebank/example.wav'
    
    # Define the path to save the enhanced audio file
    sample_output_audio_path = 'sample_enhanced.wav'
    
    # Call the remove_noise_from_audio function
    remove_noise_from_audio(sample_noisy_audio_path, sample_output_audio_path)
    
    # Load the enhanced audio file
    enhanced_audio, _ = torchaudio.load(sample_output_audio_path)
    
    # Assert that the enhanced audio file is not empty
    assert enhanced_audio.shape[0] > 0, 'The enhanced audio file is empty.'