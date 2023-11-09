def test_generate_video():
    """
    Test the generate_video function.
    """
    prompt = '(redshift style) Spider-Man is water skiing'
    video_length = 8
    height = 512
    width = 512
    num_inference_steps = 50
    guidance_scale = 7.5

    # Call the function with the test parameters
    generate_video(prompt, video_length, height, width, num_inference_steps, guidance_scale)

    # Check if the output file exists
    import os
    assert os.path.exists(f'./{prompt}.gif'), 'Video file not found.'