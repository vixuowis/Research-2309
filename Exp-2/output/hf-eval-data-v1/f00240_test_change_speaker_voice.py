def test_change_speaker_voice():
    '''
    This function tests the change_speaker_voice function.
    '''
    # Define the audio file and speaker embedding file paths
    audio_file = 'test_audio.wav'
    speaker_embedding_file = 'test_speaker_embedding.npy'
    # Call the function with the test files
    change_speaker_voice(audio_file, speaker_embedding_file)
    # Load the generated audio file
    generated_speech, _ = sf.read(audio_file)
    # Assert that the generated speech is not empty
    assert len(generated_speech) > 0