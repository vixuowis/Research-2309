def test_classify_animal_images():
    """
    This function tests the 'classify_animal_images' function by classifying a sample image of a cat.
    """
    # Define the URL of the sample image
    image_url = 'https://example.com/cat.jpg'
    
    # Classify the image
    result = classify_animal_images(image_url)
    
    # Assert that the result is 'cat'
    assert result == 'cat', f'Error: Expected cat, but got {result}'
    
    # Print a success message
    print('Test passed successfully!')

test_classify_animal_images()