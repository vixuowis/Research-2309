def test_transform_floor_plan():
    """
    Tests the transform_floor_plan function.
    """
    import os
    test_input_image_path = 'test_floor_plan.png'
    test_output_image_path = 'test_floor_plan_simplified.png'

    # Create a test input image
    Image.new('RGB', (60, 30), color = 'red').save(test_input_image_path)

    # Call the function with the test input
    transform_floor_plan(test_input_image_path, test_output_image_path)

    # Check that the output image was created
    assert os.path.exists(test_output_image_path), 'Output image was not created.'

    # Clean up test files
    os.remove(test_input_image_path)
    os.remove(test_output_image_path)

test_transform_floor_plan()