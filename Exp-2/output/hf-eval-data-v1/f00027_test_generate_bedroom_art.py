def test_generate_bedroom_art():
    """
    This function tests the 'generate_bedroom_art' function.
    It asserts that the function returns an image object.
    """
    # Call the function
    result = generate_bedroom_art()
    
    # Assert that the function returns an image object
    assert isinstance(result, Image.Image), 'The function should return an image object.'