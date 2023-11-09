def test_get_sentence_embeddings():
    # Test sentences
    sentences = ['This is an example sentence', 'Each sentence is converted']
    # Get the embeddings for the test sentences
    embeddings = get_sentence_embeddings(sentences)
    # Check the shape of the embeddings
    assert embeddings.shape == (2, 768), 'The shape of the embeddings is not correct'
    # Check the type of the embeddings
    assert isinstance(embeddings, np.ndarray), 'The type of the embeddings is not correct'

test_get_sentence_embeddings()