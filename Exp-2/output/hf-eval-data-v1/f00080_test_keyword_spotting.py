def test_keyword_spotting():
    """
    This function tests the keyword_spotting function.
    It uses a sample audio file and checks if the function returns a list.
    """
    # Define a sample audio file path
    sample_audio_file_path = 'sample_audio.wav'
    
    # Call the keyword_spotting function with the sample audio file
    detected_keywords = keyword_spotting(sample_audio_file_path)
    
    # Check if the function returns a list
    assert isinstance(detected_keywords, list), 'The function should return a list.'
    
    # Check if the list contains the correct number of elements
    assert len(detected_keywords) <= 5, 'The function should return at most 5 elements.'

test_keyword_spotting()