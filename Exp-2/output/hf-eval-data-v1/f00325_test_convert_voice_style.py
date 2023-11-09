def test_convert_voice_style():
    """
    This function tests the convert_voice_style function.
    It uses a sample audio file and a sample speaker's embeddings file.
    It asserts that the output file is correctly created and is not empty.
    """
    import os

    # Define the paths to the sample files
    sample_audio_file = 'sample_audio.wav'
    sample_speaker_embedding_file = 'sample_speaker_embedding.npy'

    # Call the function to test
    output_file = convert_voice_style(sample_audio_file, sample_speaker_embedding_file)

    # Assert that the output file is correctly created
    assert os.path.isfile(output_file), 'Output file not created'

    # Assert that the output file is not empty
    assert os.path.getsize(output_file) > 0, 'Output file is empty'

test_convert_voice_style()