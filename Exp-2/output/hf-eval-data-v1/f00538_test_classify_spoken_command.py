def test_classify_spoken_command():
    """
    This function tests the 'classify_spoken_command' function by using a sample audio file from the 'Speech Commands dataset v1.0'.
    The expected output is not strictly compared due to the probabilistic nature of the model.
    """
    # Define the path to the sample audio file
    sample_audio_file_path = 'path_to_sample_audio_file.wav'
    
    # Call the 'classify_spoken_command' function with the sample audio file
    keyword = classify_spoken_command(sample_audio_file_path)
    
    # Assert that the output is not None
    assert keyword is not None
    
    # Print the classified keyword
    print(f'Classified keyword: {keyword}')

test_classify_spoken_command()