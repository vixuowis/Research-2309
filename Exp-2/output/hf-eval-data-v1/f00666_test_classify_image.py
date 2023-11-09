def test_classify_image():
    # Test the classify_image function
    # Note: The test is not strict (does not compare numbers strictly)
    # If a dataset is provided in the performance - dataset, load the dataset, then select several samples from the dataset
    # Otherwise, use an online source

    # Test image path
    image_path = 'test_image.jpg'

    # Call the classify_image function
    result = classify_image(image_path)

    # Check the result
    assert isinstance(result, list), 'The result should be a list.'
    assert len(result) == 3, 'The result should have three elements.'
    assert all(isinstance(x, float) for x in result), 'All elements in the result should be floats.'
    assert all(0 <= x <= 1 for x in result), 'All elements in the result should be between 0 and 1.'

test_classify_image()