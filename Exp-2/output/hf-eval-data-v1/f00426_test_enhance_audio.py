def test_enhance_audio():
    """
    This function tests the 'enhance_audio' function by enhancing a sample audio file and checking
    if the output file is successfully created.
    """
    # Define the input and output file paths
    input_file = 'speechbrain/metricgan-plus-voicebank/example.wav'
    output_file = 'enhanced.wav'
    # Call the 'enhance_audio' function
    enhance_audio(input_file, output_file)
    # Check if the output file is successfully created
    assert os.path.exists(output_file), 'Output file not found.'
    # Load the output file
    enhanced, _ = torchaudio.load(output_file)
    # Check if the output file is not empty
    assert enhanced.shape[0] > 0, 'Output file is empty.'