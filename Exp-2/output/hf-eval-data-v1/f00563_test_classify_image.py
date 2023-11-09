def test_classify_image():
    """
    This function tests the 'classify_image' function with a sample image URL.
    """
    # Define a sample image URL
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    
    # Call the 'classify_image' function with the sample image URL
    predicted_class = classify_image(image_url)
    
    # Assert that the function returns a string (the predicted class)
    assert isinstance(predicted_class, str)
    
    # Print the predicted class for the sample image
    print('Predicted class for the sample image:', predicted_class)

test_classify_image()