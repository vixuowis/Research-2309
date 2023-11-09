def test_identify_street_location():
    '''
    This function tests the identify_street_location function.
    It uses a sample image and a list of possible locations.
    The test passes if the function returns a location from the list of choices.
    '''
    image_url = 'https://example.com/path-to-image.jpg'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']
    
    location = identify_street_location(image_url, choices)
    
    assert location in choices, f'Error: {location} not in {choices}'

test_identify_street_location()