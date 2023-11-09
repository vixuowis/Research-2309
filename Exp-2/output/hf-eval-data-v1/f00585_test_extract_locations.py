def test_extract_locations():
    """
    This function tests the 'extract_locations' function.
    It uses a sample text and checks if the function correctly extracts the locations from the text.
    """
    # Define the sample text
    text = 'My name is Wolfgang and I live in Berlin'
    
    # Call the function with the sample text
    locations = extract_locations(text)
    
    # Check if the function correctly extracted the location from the text
    assert 'Berlin' in locations, 'The function failed to extract the location from the text.'

test_extract_locations()