def test_classify_advertisement_video():
    '''
    Test the classify_advertisement_video function.
    '''
    # Generate a random video
    video = list(np.random.randn(8, 3, 224, 224))
    
    # Classify the video
    predicted_class = classify_advertisement_video(video)
    
    # Check that the output is a string (the class label)
    assert isinstance(predicted_class, str)