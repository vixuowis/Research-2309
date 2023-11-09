def test_generate_video_from_text():
    '''
    This function tests the generate_video_from_text function.
    It uses a sample prompt to generate a video and checks if the video file is created.
    '''
    import os
    
    # Define a sample prompt
    prompt = 'A couple sitting in a cafe and laughing while using our product'
    
    # Generate a video from the prompt
    video_path = generate_video_from_text(prompt)
    
    # Check if the video file is created
    assert os.path.exists(video_path), 'Video file not created'
    
    print('Test passed.')

test_generate_video_from_text()