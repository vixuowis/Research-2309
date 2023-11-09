def test_generate_butterfly_image():
    """
    Test function for generate_butterfly_image.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If the function does not return an image.
    """
    # Call the function
    result = generate_butterfly_image()
    # Check the result
    assert isinstance(result, np.ndarray), 'The function should return an image.'

test_generate_butterfly_image()