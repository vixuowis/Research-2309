def test_classify_video():
    """
    This function tests the classify_video function.
    It uses a sample video file for testing.
    Note: The accuracy of the model is not strictly compared due to the randomness of the model.
    """
    # Define the path to the sample video file
    video_path = 'sample_video.mp4'
    
    # Call the classify_video function with the sample video
    video_categories = classify_video(video_path)
    
    # Assert that the function returns a list
    assert isinstance(video_categories, list), 'The function should return a list.'
    
    # Assert that the list is not empty
    assert len(video_categories) > 0, 'The list should not be empty.'

test_classify_video()