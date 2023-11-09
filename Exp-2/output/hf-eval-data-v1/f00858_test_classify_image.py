def test_classify_image():
    """
    Test the classify_image function.
    """
    # Define the URL of the test image
    img_url = 'https://example.com/test_image.jpg'

    # Call the classify_image function
    result = classify_image(img_url)

    # Check that the result is a dictionary
    assert isinstance(result, dict)

    # Check that the dictionary contains the correct keys
    assert set(result.keys()) == set(['residential area', 'playground', 'stadium', 'forest', 'airport'])

    # Check that the values are probabilities (i.e., they should be between 0 and 1)
    for value in result.values():
        assert 0 <= value <= 1

test_classify_image()