def test_get_sentence_embeddings():
    """
    This function tests the get_sentence_embeddings function.
    It uses a small set of English, Italian, and Japanese sentences.
    """
    sentences = [
        'dog',
        'Cuccioli sono carini.',
        '犬と一緒にビーチを散歩するのが好き',
    ]
    embeddings = get_sentence_embeddings(sentences)
    
    assert isinstance(embeddings, torch.Tensor), 'The output should be a torch.Tensor.'
    assert embeddings.shape[0] == len(sentences), 'The number of embeddings should be equal to the number of input sentences.'

test_get_sentence_embeddings()