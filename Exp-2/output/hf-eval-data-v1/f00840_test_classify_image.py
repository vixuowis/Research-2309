def test_classify_image():
    """
    Test the classify_image function.
    """
    # Define a test image URL
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Call the classify_image function with the test image URL
    predicted_class = classify_image(test_image_url)

    # Assert that the function returns a string (the predicted class)
    assert isinstance(predicted_class, str)