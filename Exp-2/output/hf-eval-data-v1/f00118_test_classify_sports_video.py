def test_classify_sports_video():
    '''
    This function tests the classify_sports_video function.
    It generates a random video and checks if the function returns a string.
    '''
    video = list(np.random.randn(16, 3, 224, 224))
    predicted_class = classify_sports_video(video)
    assert isinstance(predicted_class, str), 'The function should return a string.'

test_classify_sports_video()