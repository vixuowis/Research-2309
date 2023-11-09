def test_generate_video_from_text():
    """
    This function tests the generate_video_from_text function.
    """
    # Define the test prompt
    test_prompt = 'A dog jumps over a fence'
    
    # Call the function with the test prompt
    video_path = generate_video_from_text(test_prompt)
    
    # Assert that the video file exists
    assert os.path.exists(video_path), 'The video file does not exist.'
    
    # Assert that the video file is not empty
    assert os.path.getsize(video_path) > 0, 'The video file is empty.'
    
    # Print a success message
    print('The test passed successfully.')

test_generate_video_from_text()