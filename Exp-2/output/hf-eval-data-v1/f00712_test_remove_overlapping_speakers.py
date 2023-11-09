def test_remove_overlapping_speakers():
    """
    This function tests the 'remove_overlapping_speakers' function.
    """
    # Define the input and output file paths
    input_file_path = 'path_to_mixed_audio.wav'
    output_file_path = 'path_to_separated_audio.wav'
    
    # Call the function
    remove_overlapping_speakers(input_file_path, output_file_path)
    
    # Read the separated audio file
    separated_audio, _ = sf.read(output_file_path)
    
    # Check that the audio is not empty
    assert separated_audio.size > 0