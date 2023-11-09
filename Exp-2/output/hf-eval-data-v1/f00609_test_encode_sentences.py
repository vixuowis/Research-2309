def test_encode_sentences():
    # Test dataset
    sentences = ['This is a test sentence.', 'This is another test sentence.']
    # Call the function with the test dataset
    embeddings = encode_sentences(sentences)
    # Assert that the function returns a list
    assert isinstance(embeddings, list), 'The function should return a list.'
    # Assert that the length of the list is equal to the number of sentences
    assert len(embeddings) == len(sentences), 'The length of the list should be equal to the number of sentences.'
    # Assert that each item in the list is a numpy array
    for embedding in embeddings:
        assert isinstance(embedding, np.ndarray), 'Each item in the list should be a numpy array.'
        # Assert that the shape of each numpy array is (768,)
        assert embedding.shape == (768,), 'The shape of each numpy array should be (768,)'

test_encode_sentences()