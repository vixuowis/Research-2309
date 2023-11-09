def test_enhance_audio():
    """
    Tests the enhance_audio function by enhancing a sample audio file and checking if the output file is created.
    """
    import os
    input_audio_file = 'test_input.wav'
    output_audio_file = 'test_output.wav'
    # Assuming test_input.wav exists in the current directory
    enhance_audio(input_audio_file, output_audio_file)
    assert os.path.exists(output_audio_file), 'Output file not created.'
    os.remove(output_audio_file)  # Cleanup after test