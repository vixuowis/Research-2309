def test_get_sentence_similarity():
    # Test sentences
    sentences = ['This is an example sentence', 'Each sentence is converted']
    # Get the similarity matrix
    similarity_matrix = get_sentence_similarity(sentences)
    # Check the shape of the similarity matrix
    assert similarity_matrix.shape == (len(sentences), len(sentences)), 'The shape of the similarity matrix is incorrect'
    # Check the diagonal elements of the similarity matrix (should be 1.0 as a sentence is always similar to itself)
    assert np.allclose(np.diag(similarity_matrix), 1.0, atol=1e-6), 'The diagonal elements of the similarity matrix are not 1.0'
test_get_sentence_similarity()