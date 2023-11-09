def test_sentence_similarity():
    # Test dataset
    test_sentences = ['This is a test sentence', 'This is another test sentence']
    # Call the function with the test dataset
    embeddings = sentence_similarity(test_sentences)
    # Assert that the embeddings are not None and have the correct shape
    assert embeddings is not None
    assert embeddings.shape == (len(test_sentences), 768)
    # Assert that the embeddings are not all zeros
    assert not np.all(embeddings == 0)

test_sentence_similarity()