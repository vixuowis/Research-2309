def test_predict_dog_breed():
    '''
    This function tests the predict_dog_breed function by comparing the output with the expected output.
    '''
    # Define the test image URL
    test_url = 'https://example.com/test_dog_image.jpg'
    
    # Call the predict_dog_breed function with the test image URL
    predicted_breed = predict_dog_breed(test_url)
    
    # Define the expected output
    expected_output = 'Golden Retriever'
    
    # Assert that the predicted breed is close to the expected output
    assert predicted_breed == expected_output, f'Expected {expected_output}, but got {predicted_breed}'

test_predict_dog_breed()