def test_is_adult():
    '''
    This function tests the is_adult function.
    It uses a sample image from the FairFace dataset.
    '''
    # Define the URL of the sample image
    url = 'https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true'
    
    # Call the is_adult function with the sample image
    result = is_adult(url)
    
    # Assert that the result is a boolean
    assert isinstance(result, bool)