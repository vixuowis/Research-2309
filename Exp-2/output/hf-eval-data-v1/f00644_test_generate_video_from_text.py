def test_generate_video_from_text():
    '''
    This function tests the generate_video_from_text function.
    '''
    # Define a test prompt
    test_prompt = 'cats playing with laser pointer'
    # Generate a video from the test prompt
    video_path = generate_video_from_text(test_prompt)
    # Assert that the video file was created
    assert os.path.exists(video_path), 'Video file was not created.'
    # Assert that the video file is not empty
    assert os.path.getsize(video_path) > 0, 'Video file is empty.'

test_generate_video_from_text()