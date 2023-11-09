def test_detect_toxic_comment():
    """
    This function tests the 'detect_toxic_comment' function with some example comments.
    """
    # Define some example comments
    comments = ['This is a great post!', 'You are an idiot.']
    
    # Expected results
    expected_results = [{'label': 'NOT_TOXIC', 'score': 0.99}, {'label': 'TOXIC', 'score': 0.99}]
    
    # Test the function with the example comments
    for i, comment in enumerate(comments):
        result = detect_toxic_comment(comment)
        
        # Check that the label is correct
        assert result['label'] == expected_results[i]['label'], f'Error: {result} != {expected_results[i]}'
        
        # Check that the score is close to the expected score
        assert abs(result['score'] - expected_results[i]['score']) < 0.01, f'Error: {result} != {expected_results[i]}'

test_detect_toxic_comment()