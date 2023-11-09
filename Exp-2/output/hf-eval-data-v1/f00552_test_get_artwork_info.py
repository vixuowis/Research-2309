def test_get_artwork_info():
    '''
    This function tests the get_artwork_info function.
    It uses a sample image and question, and asserts that the output is a string.
    '''
    # Define a sample image path and question.
    image_path = 'path/to/sample/artwork.jpg'
    question = 'What is the historical background of this artwork?'
    
    # Call the get_artwork_info function with the sample image path and question.
    answer = get_artwork_info(image_path, question)
    
    # Assert that the output is a string.
    assert isinstance(answer, str), 'The output should be a string.'
    
    print('All tests passed.')

test_get_artwork_info()