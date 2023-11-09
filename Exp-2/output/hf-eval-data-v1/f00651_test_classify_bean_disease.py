def test_classify_bean_disease():
    # Test the classify_bean_disease function
    # We will use a sample image of a bean leaf for this test
    # The image is assumed to be located in the same directory as this script
    
    # Define the path to the test image
    test_image_path = 'test_bean_leaf.jpg'
    
    # Call the classify_bean_disease function
    result = classify_bean_disease(test_image_path)
    
    # Check the result
    # Since we don't know the correct classification for the test image, we can't compare the result to a known value
    # Instead, we will just check that the result is not None
    assert result is not None, 'The function returned None'
    
    # Print the result for manual inspection
    print(f'Test image classified as: {result}')

test_classify_bean_disease()