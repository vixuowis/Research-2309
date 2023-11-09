def test_extract_text_from_manga():
    """
    This function is used to test the 'extract_text_from_manga' function.
    It uses a sample manga image and checks if the function returns a string.
    """
    # Load a sample manga image
    manga_image = 'sample_manga_image.jpg'
    
    # Call the 'extract_text_from_manga' function
    manga_text = extract_text_from_manga(manga_image)
    
    # Check if the function returns a string
    assert isinstance(manga_text, str), 'The function should return a string.'
    
    # Check if the function returns a non-empty string
    assert manga_text != '', 'The function should return a non-empty string.'

test_extract_text_from_manga()