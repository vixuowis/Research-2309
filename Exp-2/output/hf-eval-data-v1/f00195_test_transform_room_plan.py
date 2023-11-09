def test_transform_room_plan():
    '''
    This function tests the transform_room_plan function by comparing the output with an expected result.
    '''
    # Define the path of the test image
    test_image_path = 'test_room_plan.jpg'

    # Call the function with the test image
    result = transform_room_plan(test_image_path)

    # Load the expected result
    expected_result = Image.open('expected_result.jpg')

    # Compare the result with the expected result
    assert np.allclose(np.array(result), np.array(expected_result), rtol=1e-05, atol=1e-08), 'Test failed!'

    print('Test passed!')

test_transform_room_plan()