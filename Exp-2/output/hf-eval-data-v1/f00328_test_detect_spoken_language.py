def test_detect_spoken_language():
    """
    This function tests the 'detect_spoken_language' function.
    It uses a sample audio file from the VoxLingua107 development dataset.
    """
    # Define the path to the sample audio file
    sample_audio_file_path = 'https://omniglot.com/soundfiles/udhr/udhr_th.mp3'
    
    # Call the 'detect_spoken_language' function with the sample audio file
    prediction = detect_spoken_language(sample_audio_file_path)
    
    # Assert that the function returns a string (the predicted language)
    assert isinstance(prediction, str)