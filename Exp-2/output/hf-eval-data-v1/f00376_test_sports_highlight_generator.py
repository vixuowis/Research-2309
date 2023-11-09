def test_sports_highlight_generator():
    '''
    This function tests the sports_highlight_generator function.
    It uses a random video as input and checks if the output is a string (the predicted class label).
    '''
    # Generate a random video
    video = list(np.random.randn(16, 3, 224, 224))

    # Call the sports_highlight_generator function
    predicted_class = sports_highlight_generator(video)

    # Check if the output is a string
    assert isinstance(predicted_class, str), 'The output should be a string.'

    print('Test passed.')

test_sports_highlight_generator()