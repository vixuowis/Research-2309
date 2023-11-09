def test_classify_audio():
    # Test the classify_audio function
    # Use a sample audio file for testing
    audio_file_path = 'sample.wav'
    # Call the function with the sample file
    category = classify_audio(audio_file_path)
    # Check if the function returns a result
    assert category is not None, 'No result returned'
    # Check if the result is a list (as expected from the pipeline function)
    assert isinstance(category, list), 'Result is not a list'
    # Check if the list is not empty
    assert len(category) > 0, 'List is empty'
    # Check if the first item in the list is a dictionary (as expected from the pipeline function)
    assert isinstance(category[0], dict), 'First item in the list is not a dictionary'
    # Check if the dictionary has a 'label' key (as expected from the pipeline function)
    assert 'label' in category[0], 'No label in the result'
test_classify_audio()