def test_group_articles_by_topic():
    """
    This function tests the group_articles_by_topic function.
    It uses a sample dataset and asserts that the output is as expected.
    """
    # Define a sample dataset
    sentences = ['This is an example sentence', 'Each sentence is converted', 'This is another example sentence', 'Each sentence is converted differently']
    
    # Call the function with the sample dataset
    labels = group_articles_by_topic(sentences)
    
    # Assert that the output is a list
    assert isinstance(labels, list), 'Output should be a list.'
    
    # Assert that the length of the output list is equal to the length of the input list
    assert len(labels) == len(sentences), 'Output list should have the same length as the input list.'
    
    # Assert that the output list contains only integers (cluster labels)
    assert all(isinstance(label, np.int64) for label in labels), 'Output list should contain only integers.'

test_group_articles_by_topic()