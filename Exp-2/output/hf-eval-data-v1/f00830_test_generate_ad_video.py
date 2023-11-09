def test_generate_ad_video():
    """
    Test the generate_ad_video function.
    """
    test_text = 'A panda eating bamboo on a rock.'
    output_video_path = generate_ad_video(test_text)
    assert pathlib.Path(output_video_path).exists(), 'The video was not generated.'