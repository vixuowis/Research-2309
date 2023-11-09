def test_sentence_similarity():
    """
    This function tests the sentence_similarity function by comparing the output with expected results.
    The test is not strict (i.e., the similarity scores do not have to match exactly) due to the nature of the SentenceTransformer model.
    """
    # Test sentences
    sentences = ['This is an example sentence.', 'Each sentence is converted.', 'This is another similar sentence.']
    
    # Expected similarity matrix
    # Note: These are not actual expected results. Replace with actual expected results if available.
    expected_similarity_matrix = np.array([[1, 0.8, 0.9], [0.8, 1, 0.85], [0.9, 0.85, 1]])
    
    # Calculate the similarity matrix
    similarity_matrix = sentence_similarity(sentences)
    
    # Check if the calculated similarity matrix is close to the expected one
    assert np.allclose(similarity_matrix, expected_similarity_matrix, atol=0.2), 'Test failed!'
    
    print('Test passed!')

test_sentence_similarity()