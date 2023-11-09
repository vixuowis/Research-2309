def test_extract_text_from_manga():
    """
    This function tests the 'extract_text_from_manga' function.
    It uses a sample manga page image and checks if the function returns a non-empty string.
    """
    # Define the path to a sample manga page image
    sample_image_path = 'path/to/sample_manga_page.jpg'
    
    # Call the 'extract_text_from_manga' function with the sample image
    extracted_text = extract_text_from_manga(sample_image_path)
    
    # Check if the function returned a non-empty string
    assert isinstance(extracted_text, str), "The function should return a string."
    assert len(extracted_text) > 0, "The function should return a non-empty string."
    
    print("All tests passed.")

# Run the test function
test_extract_text_from_manga()