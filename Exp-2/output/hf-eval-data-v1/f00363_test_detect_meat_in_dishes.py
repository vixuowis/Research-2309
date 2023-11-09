def test_detect_meat_in_dishes():
    """
    This function tests the detect_meat_in_dishes function.
    It uses a sample image URL for testing.
    """
    # Define a sample image URL
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    
    # Call the function with the sample image URL
    result = detect_meat_in_dishes(image_url)
    
    # Assert that the result is a boolean
    assert isinstance(result, bool), 'The result should be a boolean.'
    
    # Print the result
    print(f'Meat detected: {result}')

test_detect_meat_in_dishes()