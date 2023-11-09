def test_document_similarity():
    '''
    This function tests the document_similarity function.
    It uses a small set of documents for testing.
    The expected output is a similarity matrix.
    '''
    # Test dataset
    documents = ['This is a test document.', 'This is another test document.', 'This is yet another test document.']
    
    # Call the document_similarity function
    similarity_matrix = document_similarity(documents)
    
    # Check the shape of the output matrix
    assert similarity_matrix.shape == (len(documents), len(documents)), 'The output shape is incorrect.'
    
    # Check the diagonal elements of the matrix (should be 1.0 as each document is identical to itself)
    for i in range(len(documents)):
        assert abs(similarity_matrix[i][i] - 1.0) < 1e-9, 'The diagonal elements of the similarity matrix should be 1.0.'
    
    # Check the off-diagonal elements of the matrix (should be less than 1.0 as the documents are different)
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            assert similarity_matrix[i][j] < 1.0, 'The off-diagonal elements of the similarity matrix should be less than 1.0.'
    
    print('All tests passed.')

test_document_similarity()