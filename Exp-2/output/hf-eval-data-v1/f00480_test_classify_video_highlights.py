def test_classify_video_highlights():
    """
    This function tests the classify_video_highlights function.
    It uses a sample video file and checks if the function returns a string.
    """
    video_path = 'path_to_sample_video'
    result = classify_video_highlights(video_path)
    assert isinstance(result, str), 'The function should return a string.'

test_classify_video_highlights()