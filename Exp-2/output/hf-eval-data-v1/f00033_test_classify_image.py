def test_classify_image():
    '''
    This function tests the classify_image function.
    '''
    # Define the URL of a test image and the candidate labels
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    candidate_labels = ['an image of casual dressing', 'an image of formal dressing']
    
    # Call the classify_image function
    probs = classify_image(url, candidate_labels)
    
    # Check that the output is a list
    assert isinstance(probs, list), 'Output should be a list.'
    
    # Check that the length of the output list is equal to the number of candidate labels
    assert len(probs) == len(candidate_labels), 'Output list should have the same length as the number of candidate labels.'
    
    # Check that the sum of the probabilities is approximately 1
    assert abs(sum(probs) - 1) < 0.01, 'The sum of the probabilities should be approximately 1.'

test_classify_image()