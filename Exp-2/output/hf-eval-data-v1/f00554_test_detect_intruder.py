def test_detect_intruder():
    # Define the path to the test image
    test_image_path = 'test_image.jpg'

    # Call the function with the test image
    result = detect_intruder(test_image_path)

    # Assert that the result is a string (since the function should return a string answer)
    assert isinstance(result, str)

    # Print the result
    print(result)

# Call the test function
test_detect_intruder()