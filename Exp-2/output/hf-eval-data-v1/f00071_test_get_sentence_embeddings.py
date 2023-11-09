def test_get_sentence_embeddings():
    '''
    This function tests the 'get_sentence_embeddings' function.
    It uses a sample dataset and asserts that the output is as expected.
    '''
    # Test dataset
    sentences = ['This is an example sentence', 'Each sentence is converted']
    # Get sentence embeddings
    embeddings = get_sentence_embeddings(sentences)
    # Assert that the embeddings are not None and their shape is as expected
    assert embeddings is not None
    assert embeddings.shape == (len(sentences), 768)

test_get_sentence_embeddings()