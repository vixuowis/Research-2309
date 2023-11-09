def test_generate_spiderman_surfing_video():
    """
    This function tests the generate_spiderman_surfing_video function.
    It asserts that the output video file exists.
    """
    # Call the function to test
    video_path = generate_spiderman_surfing_video()
    
    # Assert that the video file exists
    assert os.path.exists(video_path), 'Video file does not exist.'