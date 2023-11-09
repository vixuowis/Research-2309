def test_enhance_audio():
    '''
    This function tests the enhance_audio function by using a sample audio file.
    '''
    # Define the input and output audio files
    input_audio_file = 'speechbrain/sepformer-whamr-enhancement/example_whamr.wav'
    output_audio_file = 'enhanced_whamr.wav'
    # Call the enhance_audio function
    enhance_audio(input_audio_file, output_audio_file)
    # Load the enhanced audio file
    enhanced_audio, _ = torchaudio.load(output_audio_file)
    # Assert that the enhanced audio is not None
    assert enhanced_audio is not None