def test_extract_insurance_info():
    """
    This function tests the 'extract_insurance_info' function by using a sample insurance policy document image.
    It asserts that the returned answers are not None.
    """
    # Use a sample insurance policy document image for testing
    image_path = 'path/to/sample/image.jpg'

    # Call the function with the sample image
    answers = extract_insurance_info(image_path)

    # Assert that the returned answers are not None
    for answer in answers.values():
        assert answer is not None, 'The extracted answer should not be None.'

    print('All tests passed.')

test_extract_insurance_info()