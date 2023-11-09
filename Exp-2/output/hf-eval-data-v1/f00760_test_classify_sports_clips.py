def test_classify_sports_clips():
    """
    Test the function classify_sports_clips.
    """
    video = list(np.random.randn(8, 3, 224, 224))  # Test with random video data
    predicted_class = classify_sports_clips(video)
    assert isinstance(predicted_class, str), 'The output should be a string.'

test_classify_sports_clips()