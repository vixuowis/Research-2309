def test_convert_voice():
    '''
    This function tests the convert_voice function.
    It uses a sample audio file and a sample speaker embedding file.
    
    Returns:
    None
    '''
    # Define the paths to the sample files
    input_audio_path = 'sample_audio.wav'
    speaker_embedding_path = 'sample_speaker_embedding.npy'
    output_audio_path = 'sample_output_audio.wav'
    
    # Call the function with the sample files
    convert_voice(input_audio_path, speaker_embedding_path, output_audio_path)
    
    # Load the output audio file
    output_audio, _ = sf.read(output_audio_path)
    
    # Check that the output audio file is not empty
    assert len(output_audio) > 0, 'The output audio file is empty.'
    
    # Check that the output audio file has the correct sample rate
    assert sf.info(output_audio_path).samplerate == 16000, 'The output audio file has the wrong sample rate.'

test_convert_voice()