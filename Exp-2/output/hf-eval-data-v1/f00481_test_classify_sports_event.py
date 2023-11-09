def test_classify_sports_event():
    '''
    Function to test the classify_sports_event function.
    It uses a random video data for the test.
    '''
    
    # Generate random video data
    video = list(np.random.randn(16, 3, 224, 224))
    
    # Call the function with the test data
    result = classify_sports_event(video)
    
    # Since we don't have the actual class for the random data, we can only check if the result is not None
    assert result is not None, 'The function did not return a result'
    
    print('Test passed.')

# Run the test function
test_classify_sports_event()