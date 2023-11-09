def test_generate_sentence_embeddings():
    """
    This function tests the 'generate_sentence_embeddings' function.
    It uses a sample sentence in Russian and checks if the output is a tensor.
    """
    sentences = ['Анализировать текст российской газеты']
    sentence_embeddings = generate_sentence_embeddings(sentences)
    assert isinstance(sentence_embeddings, torch.Tensor), 'The output should be a tensor.'

test_generate_sentence_embeddings()