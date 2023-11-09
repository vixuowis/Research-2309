def test_video_classification():
    '''
    Function to test the video_classification function.
    
    The function generates a random video and uses the video_classification function to classify it.
    The output is then checked to ensure it is a torch.Tensor.
    '''
    video = list(np.random.randn(16, 3, 224, 224))
    output = video_classification(video)
    assert isinstance(output, torch.Tensor), 'Output should be a torch.Tensor.'

test_video_classification()