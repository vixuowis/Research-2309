def test_detect_tables_in_image():
    """
    This function tests the detect_tables_in_image function.
    It uses a sample image from an online source.
    """
    image_url = 'https://example.com/sample_image.jpg'
    # Download the image
    response = requests.get(image_url)
    image_path = 'sample_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(response.content)
    
    detected_tables = detect_tables_in_image(image_path)
    
    # Check if the function returns a list
    assert isinstance(detected_tables, list)
    
    # Check if each item in the list is a tuple with three elements
    for table in detected_tables:
        assert isinstance(table, tuple)
        assert len(table) == 3
        
    # Check if the first element of each tuple is a string (the label)
    # Check if the second element of each tuple is a float (the confidence score)
    # Check if the third element of each tuple is a list (the location)
    for label, score, location in detected_tables:
        assert isinstance(label, str)
        assert isinstance(score, float)
        assert isinstance(location, list)
        
    # Remove the downloaded image
    os.remove(image_path)

test_detect_tables_in_image()