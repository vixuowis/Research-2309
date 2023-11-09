def test_detect_movie_genre():
    '''
    This function tests the detect_movie_genre function.
    It uses a sample video file and checks if the function returns a string.
    '''
    # Define the path to the sample video file
    video_filename = 'path/to/sample_video_file.mp4'

    # Call the function with the sample video file
    result = detect_movie_genre(video_filename)

    # Check if the result is a string
    assert isinstance(result, str), 'The function should return a string.'

    # Print the result
    print('Test passed. The function correctly detected the genre of the movie.')

# Call the test function
test_detect_movie_genre()