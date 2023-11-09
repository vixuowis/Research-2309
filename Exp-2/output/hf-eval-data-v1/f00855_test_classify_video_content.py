def test_classify_video_content():
    '''
    Test the function classify_video_content.
    
    '''
    video = list(np.random.randn(16, 3, 224, 224))
    category = classify_video_content(video)
    assert isinstance(category, str), 'The output should be a string representing the category.'

test_classify_video_content()