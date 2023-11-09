def test_video_classification():
    '''
    This function tests the video_classification function.
    It generates a random video and checks if the function returns a string.
    '''
    video = list(np.random.randn(16, 3, 224, 224))
    predicted_class = video_classification(video)
    assert isinstance(predicted_class, str), 'The function should return a string.'

test_video_classification()