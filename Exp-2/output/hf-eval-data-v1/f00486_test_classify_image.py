def test_classify_image():
    """
    This function tests the 'classify_image' function with a sample image.
    """
    # Define the path to the sample image
    image_path = './path/to/sample/image.jpg'
    
    # Call the 'classify_image' function
    result = classify_image(image_path)
    
    # Define the expected categories
    expected_categories = ['landscape', 'cityscape', 'beach', 'forest', 'animals']
    
    # Assert that the result is in the expected categories
    assert result in expected_categories, f'Error: {result} not in {expected_categories}'
    
    print('Test passed.')

# Call the test function
test_classify_image()