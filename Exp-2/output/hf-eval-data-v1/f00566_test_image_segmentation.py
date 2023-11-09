def test_image_segmentation():
    # Define a test image URL
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # Call the image_segmentation function with the test image URL
    result = image_segmentation(test_image_url)
    # Assert that the result is not None
    assert result is not None
    # Assert that the result is a numpy array (as the segmentation map is expected to be a numpy array)
    assert isinstance(result, np.ndarray)

test_image_segmentation()