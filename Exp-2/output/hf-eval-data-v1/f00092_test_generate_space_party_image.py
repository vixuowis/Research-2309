def test_generate_space_party_image():
    """
    This function tests the generate_space_party_image function.
    It asserts that the function does not raise an exception and that the output image file exists.
    """
    import os

    # Call the function
    try:
        generate_space_party_image()
    except Exception as e:
        assert False, f"Function failed with error: {e}"

    # Check that the output file exists
    assert os.path.exists('space_party.png'), "Output image file does not exist"