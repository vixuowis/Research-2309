def test_classify_computer_parts():
    '''
    This function tests the classify_computer_parts function by using a sample image of a computer part.
    '''
    # Define the file path of the sample image
    sample_image_file_path = 'sample_computer_part.jpg'
    # Call the classify_computer_parts function with the sample image
    predicted_label = classify_computer_parts(sample_image_file_path)
    # Assert that the function returns a string (the predicted label)
    assert isinstance(predicted_label, str)