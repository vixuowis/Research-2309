def test_classify_text():
    '''
    This function tests the classify_text function by providing a sample text message and candidate labels.
    '''
    # Define a sample text message and candidate labels
    text_message = 'Your monthly bank statement is now available.'
    candidate_labels = ['finances', 'health', 'entertainment']
    
    # Call the classify_text function
    classification_result = classify_text(text_message, candidate_labels)
    
    # Assert that the function returns a dictionary
    assert isinstance(classification_result, dict)
    
    # Assert that the dictionary contains the 'labels' and 'scores' keys
    assert 'labels' in classification_result
    assert 'scores' in classification_result
    
    # Assert that the most likely label is 'finances'
    assert classification_result['labels'][0] == 'finances'

test_classify_text()