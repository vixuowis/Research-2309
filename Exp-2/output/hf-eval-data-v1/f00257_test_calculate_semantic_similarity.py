def test_calculate_semantic_similarity():
    """
    This function tests the calculate_semantic_similarity function.
    It uses assert to verify the function's output.
    """
    # Test data
    text1 = 'The cat sat on the mat'
    text2 = 'The cat is sitting on the mat'
    text3 = 'The dog is in the garden'
    
    # Calculate the semantic similarities
    sim1 = calculate_semantic_similarity(text1, text2)
    sim2 = calculate_semantic_similarity(text1, text3)
    
    # Assert that the semantic similarity between text1 and text2 is greater than the semantic similarity between text1 and text3
    assert sim1 > sim2, 'Test failed: The semantic similarity between text1 and text2 should be greater than the semantic similarity between text1 and text3'
    
    print('All tests passed.')

# Run the test function
test_calculate_semantic_similarity()