# Test function for classify_image
# @param: None
# @return: None
def test_classify_image():
    # Define a test image URL
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # Call the classify_image function with the test image URL
    predicted_class = classify_image(test_image_url)
    # Assert that the predicted class is not None
    assert predicted_class is not None, 'The predicted class should not be None.'
    # Print the predicted class
    print('Predicted class:', predicted_class)

# Call the test function
test_classify_image()