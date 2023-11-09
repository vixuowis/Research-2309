def test_generate_video_from_text():
    '''
    This function tests the generate_video_from_text function.
    It uses a sample prompt and checks if a video is generated and saved at the returned path.
    '''
    # Sample prompt
    prompt = 'Spiderman is surfing'
    # Call the function with the sample prompt
    video_path = generate_video_from_text(prompt)
    # Check if a file exists at the returned path
    assert os.path.isfile(video_path), 'No video file found at the returned path.'