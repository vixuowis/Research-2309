def test_emotion_recognition():
    '''
    This function tests the emotion_recognition function by using a sample from the 'anton-l/superb_demo' dataset.
    '''
    # Load the dataset
    dataset = load_dataset('anton-l/superb_demo', 'er', split='session1')
    
    # Select a sample from the dataset
    sample = dataset[0]['file']
    
    # Test the emotion_recognition function
    labels = emotion_recognition(sample, top_k=5)
    
    # Check that the function returns the correct number of predictions
    assert len(labels) == 5, 'The function should return the top 5 predictions.'
    
    # Check that each prediction is a dictionary with the correct keys
    for label in labels:
        assert 'score' in label, 'Each prediction should have a score.'
        assert 'label' in label, 'Each prediction should have a label.'

test_emotion_recognition()